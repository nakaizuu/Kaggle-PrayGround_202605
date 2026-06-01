from pathlib import Path
import copy
import gc
import json
import random
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Required package is missing. Install dependencies with:\n"
        "pip install torch scikit-learn pandas numpy"
    ) from exc


warnings.filterwarnings("ignore")

LOCAL_DATA_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
KAGGLE_DATA_DIR = Path("/kaggle/input/datasets/mizukinakaizuuu/input-4data")
KAGGLE_LIGHTGBM_OOF_DIRS = [
    Path("/kaggle/input/datasets/mizukinakaizuuu/7thResult"),
    Path("/kaggle/input/datasets/mizukinakaizuuu/7thresult"),
]

if (KAGGLE_DATA_DIR / "train.csv").exists():
    INPUT_DIR = KAGGLE_DATA_DIR
    WORK_DIR = Path("/kaggle/working")
else:
    INPUT_DIR = LOCAL_DATA_DIR
    WORK_DIR = LOCAL_DATA_DIR

TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"
SAMPLE_PATH = INPUT_DIR / "sample_submission.csv"
EXTERNAL_PATH = INPUT_DIR / "f1_strategy_dataset_v4.csv"
LIGHTGBM_OOF_CANDIDATES = [
    *[
        oof_dir / filename
        for oof_dir in KAGGLE_LIGHTGBM_OOF_DIRS
        for filename in [
            "7th_lightgbm_oof_predictions.csv",
            "oof_predictions.csv",
        ]
    ],
    LOCAL_DATA_DIR
    / "outputs"
    / "7th_external_data_seed_ensemble"
    / "oof_predictions.csv",
]
OUTPUT_PATH = WORK_DIR / "submission_pytorch_tabular_mlp.csv"
DIAGNOSTIC_DIR = WORK_DIR / "outputs" / "9th_pytorch_tabular_mlp"

ID_COL = "id"
TARGET_COL = "PitNextLap"
N_SPLITS = 5
INNER_TE_SPLITS = 5
SEED = 42
EXTERNAL_WEIGHT = 0.65
TE_SMOOTHING = 30.0

# This first PyTorch experiment is deliberately smaller than the reference
# notebook. Increase EPOCHS or HIDDEN_DIMS later only if the baseline is useful.
EPOCHS = 5
BATCH_SIZE = 1024
PREDICT_BATCH_SIZE = 8192
HIDDEN_DIMS = [256, 256, 128]
EMBEDDING_DIM_CAP = 16
DROPOUT = 0.10
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 5e-4
POS_WEIGHT_POWER = 0.5

RAW_NUMERIC_COLS = [
    "Year",
    "PitStop",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]

BASE_CATEGORICAL_COLS = [
    "Driver",
    "Compound",
    "Race",
]

DERIVED_CATEGORICAL_COLS = [
    "Year_cat",
    "PitStop_cat",
    "LapNumber_cat",
    "Stint_cat",
    "TyreLife_cat",
    "Position_cat",
    "RaceProgress_bin",
    "LapTime_bin",
    "Race__Compound",
    "Race__Year",
]

TARGET_ENCODING_COLS = [
    "Race__Compound",
    "Race__Year",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    external = pd.read_csv(EXTERNAL_PATH)

    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in train.csv")
    if TARGET_COL not in external.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in external data")
    if ID_COL not in sample.columns:
        raise ValueError(f"ID column '{ID_COL}' was not found in sample_submission.csv")

    # This column cannot be created reliably for the competition test rows.
    external = external.drop(columns=["Normalized_TyreLife"], errors="ignore")

    return train, test, sample, external


def check_required_columns(frame, frame_name):
    required = set(RAW_NUMERIC_COLS + BASE_CATEGORICAL_COLS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {missing}")


def make_base_features(raw):
    check_required_columns(raw, "Input data")
    features = pd.DataFrame(index=raw.index)

    for col in RAW_NUMERIC_COLS:
        features[col] = pd.to_numeric(raw[col], errors="coerce").astype("float32")

    lap_number = features["LapNumber"].clip(lower=1.0)
    race_progress = features["RaceProgress"].clip(lower=1e-6)
    degradation_abs = features["Cumulative_Degradation"].abs()

    # Interactions adapted from the public RealMLP notebook.
    features["LapNumberPerRaceProgress"] = features["LapNumber"] / race_progress
    features["TyreLifePerLapNumber"] = features["TyreLife"] / lap_number
    features["LapTimeXDegradation"] = (
        features["LapTime (s)"] * features["Cumulative_Degradation"]
    )
    features["LapTimeXAbsDegradation"] = features["LapTime (s)"] * degradation_abs
    features["LapTimePerAbsDegradation"] = (
        features["LapTime (s)"] / (degradation_abs + 1e-6)
    )

    for col in BASE_CATEGORICAL_COLS:
        features[col] = raw[col].astype("string").fillna("__MISSING__")

    features["Year_cat"] = np.floor(features["Year"]).astype("Int64").astype("string")
    features["PitStop_cat"] = np.floor(features["PitStop"]).astype("Int64").astype("string")
    features["LapNumber_cat"] = np.floor(features["LapNumber"]).astype("Int64").astype("string")
    features["Stint_cat"] = np.floor(features["Stint"]).astype("Int64").astype("string")
    features["TyreLife_cat"] = np.floor(features["TyreLife"]).astype("Int64").astype("string")
    features["Position_cat"] = np.floor(features["Position"]).astype("Int64").astype("string")
    features["RaceProgress_bin"] = (
        np.floor(features["RaceProgress"] * 100.0).astype("Int64").astype("string")
    )
    features["LapTime_bin"] = (
        np.floor(features["LapTime (s)"]).astype("Int64").astype("string")
    )
    features["Race__Compound"] = features["Race"] + "__" + features["Compound"]
    features["Race__Year"] = features["Race"] + "__" + features["Year_cat"]

    return features


def add_count_features(train_features, test_features, external_features):
    combined = pd.concat(
        [train_features, test_features, external_features],
        ignore_index=True,
    )
    count_cols = [
        "Driver",
        "Compound",
        "Race",
        "Race__Compound",
        "Race__Year",
    ]

    for col in count_cols:
        count_map = combined[col].value_counts(dropna=False)
        count_name = f"{col}_count"
        train_features[count_name] = train_features[col].map(count_map).fillna(0).astype("float32")
        test_features[count_name] = test_features[col].map(count_map).fillna(0).astype("float32")
        external_features[count_name] = (
            external_features[col].map(count_map).fillna(0).astype("float32")
        )

    return train_features, test_features, external_features


def build_category_maps(train_features, test_features, external_features, categorical_cols):
    combined = pd.concat(
        [
            train_features[categorical_cols],
            test_features[categorical_cols],
            external_features[categorical_cols],
        ],
        ignore_index=True,
    )
    category_maps = {}

    for col in categorical_cols:
        values = combined[col].astype("string").fillna("__MISSING__").unique()
        category_maps[col] = {value: index + 1 for index, value in enumerate(values)}

    return category_maps


def encode_categories(frame, categorical_cols, category_maps):
    encoded = np.zeros((len(frame), len(categorical_cols)), dtype=np.int64)

    for index, col in enumerate(categorical_cols):
        encoded[:, index] = (
            frame[col]
            .astype("string")
            .fillna("__MISSING__")
            .map(category_maps[col])
            .fillna(0)
            .astype("int64")
            .to_numpy()
        )

    return encoded


def weighted_target_encoding_map(keys, target, sample_weight):
    table = pd.DataFrame(
        {
            "key": keys.astype("string").fillna("__MISSING__").to_numpy(),
            "target": np.asarray(target, dtype=float),
            "weight": np.asarray(sample_weight, dtype=float),
        }
    )
    table["weighted_target"] = table["target"] * table["weight"]
    global_mean = table["weighted_target"].sum() / table["weight"].sum()
    grouped = table.groupby("key", dropna=False).agg(
        weighted_target_sum=("weighted_target", "sum"),
        weight_sum=("weight", "sum"),
    )
    encoded = (
        grouped["weighted_target_sum"] + TE_SMOOTHING * global_mean
    ) / (grouped["weight_sum"] + TE_SMOOTHING)

    return encoded, global_mean


def apply_target_encoding(keys, encoding_map, fallback):
    return (
        keys.astype("string")
        .fillna("__MISSING__")
        .map(encoding_map)
        .fillna(fallback)
        .astype("float32")
        .to_numpy()
    )


def add_fold_target_encoding(
    train_features,
    train_target,
    train_weight,
    valid_features,
    test_features,
    seed,
):
    train_features = train_features.copy()
    valid_features = valid_features.copy()
    test_features = test_features.copy()

    inner_folds = StratifiedKFold(
        n_splits=INNER_TE_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    train_target_array = np.asarray(train_target, dtype=int)
    train_weight_array = np.asarray(train_weight, dtype=float)

    for col in TARGET_ENCODING_COLS:
        encoded_train = np.zeros(len(train_features), dtype=np.float32)

        for inner_train_idx, inner_valid_idx in inner_folds.split(
            train_features,
            train_target_array,
        ):
            encoding_map, fallback = weighted_target_encoding_map(
                train_features.iloc[inner_train_idx][col],
                train_target_array[inner_train_idx],
                train_weight_array[inner_train_idx],
            )
            encoded_train[inner_valid_idx] = apply_target_encoding(
                train_features.iloc[inner_valid_idx][col],
                encoding_map,
                fallback,
            )

        full_map, full_fallback = weighted_target_encoding_map(
            train_features[col],
            train_target_array,
            train_weight_array,
        )
        te_name = f"{col}_target_encoding"
        train_features[te_name] = encoded_train
        valid_features[te_name] = apply_target_encoding(
            valid_features[col],
            full_map,
            full_fallback,
        )
        test_features[te_name] = apply_target_encoding(
            test_features[col],
            full_map,
            full_fallback,
        )

    return train_features, valid_features, test_features


def make_numeric_arrays(train_features, valid_features, test_features, numeric_cols):
    train_numeric = (
        train_features[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
        .to_numpy()
    )
    valid_numeric = (
        valid_features[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
        .to_numpy()
    )
    test_numeric = (
        test_features[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
        .to_numpy()
    )

    medians = np.nanmedian(train_numeric, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_numeric = np.where(np.isnan(train_numeric), medians, train_numeric)
    valid_numeric = np.where(np.isnan(valid_numeric), medians, valid_numeric)
    test_numeric = np.where(np.isnan(test_numeric), medians, test_numeric)

    means = train_numeric.mean(axis=0)
    stds = train_numeric.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)

    train_numeric = np.clip((train_numeric - means) / stds, -10.0, 10.0)
    valid_numeric = np.clip((valid_numeric - means) / stds, -10.0, 10.0)
    test_numeric = np.clip((test_numeric - means) / stds, -10.0, 10.0)

    return (
        train_numeric.astype("float32"),
        valid_numeric.astype("float32"),
        test_numeric.astype("float32"),
    )


class TabularMLP(nn.Module):
    def __init__(self, num_numeric, category_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList()
        embedding_dims = []

        for category_size in category_sizes:
            embedding_dim = min(EMBEDDING_DIM_CAP, max(4, int(np.sqrt(category_size))))
            self.embeddings.append(
                nn.Embedding(category_size, embedding_dim, padding_idx=0)
            )
            embedding_dims.append(embedding_dim)

        input_dim = num_numeric + sum(embedding_dims)
        layers = []

        for hidden_dim in HIDDEN_DIMS:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(DROPOUT),
                ]
            )
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, numeric, categorical):
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        combined = torch.cat([numeric] + embedded, dim=1)
        return self.output(self.network(combined)).squeeze(1)


def make_loader(numeric, categorical, target, sample_weight, shuffle, device):
    dataset = TensorDataset(
        torch.from_numpy(numeric),
        torch.from_numpy(categorical),
        torch.from_numpy(np.asarray(target, dtype=np.float32)),
        torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def predict(model, numeric, categorical, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(numeric), PREDICT_BATCH_SIZE):
            end = start + PREDICT_BATCH_SIZE
            numeric_batch = torch.from_numpy(numeric[start:end]).to(device)
            categorical_batch = torch.from_numpy(categorical[start:end]).to(device)
            logits = model(numeric_batch, categorical_batch)
            predictions.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(predictions)


def train_fold(
    fold,
    train_numeric,
    train_categorical,
    train_target,
    train_weight,
    valid_numeric,
    valid_categorical,
    valid_target,
    test_numeric,
    test_categorical,
    category_sizes,
    device,
):
    seed_everything(SEED + fold)
    model = TabularMLP(train_numeric.shape[1], category_sizes).to(device)
    train_loader = make_loader(
        train_numeric,
        train_categorical,
        train_target,
        train_weight,
        shuffle=True,
        device=device,
    )

    weighted_positive = np.asarray(train_weight)[np.asarray(train_target) == 1].sum()
    weighted_negative = np.asarray(train_weight)[np.asarray(train_target) == 0].sum()
    pos_weight_value = (weighted_negative / weighted_positive) ** POS_WEIGHT_POWER
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_auc = -np.inf
    best_epoch = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen_rows = 0

        for numeric_batch, categorical_batch, target_batch, weight_batch in train_loader:
            numeric_batch = numeric_batch.to(device, non_blocking=True)
            categorical_batch = categorical_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            weight_batch = weight_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric_batch, categorical_batch)
            row_loss = loss_function(logits, target_batch)
            loss = (row_loss * weight_batch).sum() / weight_batch.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(target_batch)
            seen_rows += len(target_batch)

        scheduler.step()
        valid_prediction = predict(model, valid_numeric, valid_categorical, device)
        valid_auc = roc_auc_score(valid_target, valid_prediction)
        print(
            f"Fold {fold} Epoch {epoch}/{EPOCHS}: "
            f"loss={running_loss / seen_rows:.6f}, AUC={valid_auc:.6f}"
        )

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    valid_prediction = predict(model, valid_numeric, valid_categorical, device)
    test_prediction = predict(model, test_numeric, test_categorical, device)
    print(f"Fold {fold}: best AUC={best_auc:.6f} at epoch {best_epoch}")

    return valid_prediction, test_prediction, best_auc, best_epoch, pos_weight_value


def compare_with_lightgbm(train, pytorch_oof):
    lightgbm_oof_path = next(
        (path for path in LIGHTGBM_OOF_CANDIDATES if path.exists()),
        None,
    )
    if lightgbm_oof_path is None:
        return None

    lightgbm_oof = pd.read_csv(lightgbm_oof_path)
    if ID_COL not in lightgbm_oof.columns or "oof_prediction" not in lightgbm_oof.columns:
        return None

    comparison = pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET_COL: train[TARGET_COL],
            "pytorch_oof": pytorch_oof,
        }
    ).merge(
        lightgbm_oof[[ID_COL, "oof_prediction"]],
        on=ID_COL,
        how="inner",
    )

    if len(comparison) != len(train):
        return None

    pearson = comparison["pytorch_oof"].corr(comparison["oof_prediction"])
    spearman = comparison["pytorch_oof"].rank().corr(
        comparison["oof_prediction"].rank()
    )
    blend_records = []

    for lightgbm_weight in np.arange(0.0, 1.01, 0.1):
        pytorch_weight = 1.0 - lightgbm_weight
        probability_blend = (
            lightgbm_weight * comparison["oof_prediction"]
            + pytorch_weight * comparison["pytorch_oof"]
        )
        rank_blend = (
            lightgbm_weight * comparison["oof_prediction"].rank(pct=True)
            + pytorch_weight * comparison["pytorch_oof"].rank(pct=True)
        )
        blend_records.append(
            {
                "lightgbm_weight": lightgbm_weight,
                "pytorch_weight": pytorch_weight,
                "probability_blend_auc": roc_auc_score(
                    comparison[TARGET_COL],
                    probability_blend,
                ),
                "rank_blend_auc": roc_auc_score(
                    comparison[TARGET_COL],
                    rank_blend,
                ),
            }
        )

    return {
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "blend_metrics": pd.DataFrame(blend_records),
    }


def save_submission(sample, test, prediction):
    submission = sample.copy()
    if ID_COL in test.columns:
        submission[ID_COL] = test[ID_COL].values
    submission[TARGET_COL] = np.clip(prediction, 0.0, 1.0)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved submission: {OUTPUT_PATH}")
    print(submission.head())


def main():
    seed_everything(SEED)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, test, sample, external = load_data()
    train_features = make_base_features(train)
    test_features = make_base_features(test)
    external_features = make_base_features(external)
    train_features, test_features, external_features = add_count_features(
        train_features,
        test_features,
        external_features,
    )

    categorical_cols = BASE_CATEGORICAL_COLS + DERIVED_CATEGORICAL_COLS
    category_maps = build_category_maps(
        train_features,
        test_features,
        external_features,
        categorical_cols,
    )
    category_sizes = [len(category_maps[col]) + 1 for col in categorical_cols]
    numeric_cols = [
        col for col in train_features.columns if col not in categorical_cols
    ] + [f"{col}_target_encoding" for col in TARGET_ENCODING_COLS]

    y = train[TARGET_COL].astype(int).reset_index(drop=True)
    y_external = external[TARGET_COL].astype(int).reset_index(drop=True)
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_prediction = np.zeros(len(train), dtype=float)
    test_prediction = np.zeros(len(test), dtype=float)
    fold_records = []

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {WORK_DIR}")
    if device.type == "cpu":
        print("Warning: CUDA was not detected. Training will be much slower on CPU.")
    print(f"Competition train shape: {train.shape}")
    print(f"External train shape: {external.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"External weight: {EXTERNAL_WEIGHT}")

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_features, y), start=1):
        print("\n" + "=" * 60)
        print(f"Fold {fold}/{N_SPLITS}")
        print("=" * 60)

        fold_train_features = pd.concat(
            [train_features.iloc[train_idx], external_features],
            ignore_index=True,
        )
        fold_train_target = pd.concat(
            [y.iloc[train_idx], y_external],
            ignore_index=True,
        )
        fold_train_weight = np.concatenate(
            [
                np.ones(len(train_idx), dtype=np.float32),
                np.full(len(external_features), EXTERNAL_WEIGHT, dtype=np.float32),
            ]
        )
        fold_valid_features = train_features.iloc[valid_idx].copy()
        fold_test_features = test_features.copy()
        fold_train_features, fold_valid_features, fold_test_features = (
            add_fold_target_encoding(
                fold_train_features,
                fold_train_target,
                fold_train_weight,
                fold_valid_features,
                fold_test_features,
                seed=SEED + fold,
            )
        )

        train_numeric, valid_numeric, test_numeric = make_numeric_arrays(
            fold_train_features,
            fold_valid_features,
            fold_test_features,
            numeric_cols,
        )
        train_categorical = encode_categories(
            fold_train_features,
            categorical_cols,
            category_maps,
        )
        valid_categorical = encode_categories(
            fold_valid_features,
            categorical_cols,
            category_maps,
        )
        test_categorical = encode_categories(
            fold_test_features,
            categorical_cols,
            category_maps,
        )

        (
            fold_valid_prediction,
            fold_test_prediction,
            fold_auc,
            best_epoch,
            pos_weight,
        ) = train_fold(
            fold,
            train_numeric,
            train_categorical,
            fold_train_target.to_numpy(),
            fold_train_weight,
            valid_numeric,
            valid_categorical,
            y.iloc[valid_idx].to_numpy(),
            test_numeric,
            test_categorical,
            category_sizes,
            device,
        )
        oof_prediction[valid_idx] = fold_valid_prediction
        test_prediction += fold_test_prediction / N_SPLITS
        fold_records.append(
            {
                "fold": fold,
                "auc": fold_auc,
                "best_epoch": best_epoch,
                "pos_weight": pos_weight,
                "competition_train_rows": len(train_idx),
                "external_train_rows": len(external_features),
                "external_weight": EXTERNAL_WEIGHT,
            }
        )

        del (
            fold_train_features,
            fold_valid_features,
            fold_test_features,
            train_numeric,
            valid_numeric,
            test_numeric,
            train_categorical,
            valid_categorical,
            test_categorical,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_auc = roc_auc_score(y, oof_prediction)
    fold_metrics = pd.DataFrame(fold_records)
    comparison = compare_with_lightgbm(train, oof_prediction)

    print("\n" + "=" * 60)
    print("Pre-submission diagnostics")
    print("=" * 60)
    print(f"PyTorch OOF AUC: {ensemble_auc:.6f}")
    print(
        "Fold AUC: "
        f"mean={fold_metrics['auc'].mean():.6f}, "
        f"std={fold_metrics['auc'].std():.6f}, "
        f"min={fold_metrics['auc'].min():.6f}, "
        f"max={fold_metrics['auc'].max():.6f}"
    )
    print("\nTest prediction distribution:")
    print(pd.Series(test_prediction, name=TARGET_COL).describe().to_string())

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET_COL: y,
            "oof_prediction": oof_prediction,
        }
    ).to_csv(DIAGNOSTIC_DIR / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(DIAGNOSTIC_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "pytorch_oof_auc",
                "competition_train_rows",
                "external_train_rows",
                "external_weight",
                "device",
            ],
            "value": [
                ensemble_auc,
                len(train),
                len(external),
                EXTERNAL_WEIGHT,
                str(device),
            ],
        }
    ).to_csv(DIAGNOSTIC_DIR / "run_summary.csv", index=False)

    with (DIAGNOSTIC_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "n_splits": N_SPLITS,
                "inner_te_splits": INNER_TE_SPLITS,
                "seed": SEED,
                "external_weight": EXTERNAL_WEIGHT,
                "te_smoothing": TE_SMOOTHING,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "hidden_dims": HIDDEN_DIMS,
                "embedding_dim_cap": EMBEDDING_DIM_CAP,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "pos_weight_power": POS_WEIGHT_POWER,
            },
            file,
            indent=2,
        )

    if comparison is not None:
        print("\nComparison with 7th LightGBM OOF:")
        print(f"Pearson correlation:  {comparison['pearson_correlation']:.6f}")
        print(f"Spearman correlation: {comparison['spearman_correlation']:.6f}")
        print("\nOOF blend candidates:")
        print(comparison["blend_metrics"].to_string(index=False))
        comparison["blend_metrics"].to_csv(
            DIAGNOSTIC_DIR / "lightgbm_blend_metrics.csv",
            index=False,
        )
    else:
        print(
            "\nSkipped LightGBM blend diagnostics. "
            "Add oof_predictions.csv to the 7thResult dataset "
            "to enable them."
        )

    save_submission(sample, test, test_prediction)
    print(f"\nSaved diagnostics: {DIAGNOSTIC_DIR}")


if __name__ == "__main__":
    main()
