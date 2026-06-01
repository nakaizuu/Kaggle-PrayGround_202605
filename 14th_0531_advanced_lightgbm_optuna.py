from pathlib import Path
import gc
import json
import warnings

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Required package is missing. Install dependencies with:\n"
        "pip install lightgbm optuna scikit-learn pandas numpy"
    ) from exc


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

LOCAL_DATA_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
KAGGLE_DATA_DIR = Path("/kaggle/input/datasets/mizukinakaizuuu/input-4data")

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
OUTPUT_PATH = WORK_DIR / "submission_14th_advanced_lightgbm_optuna.csv"
DIAGNOSTIC_DIR = WORK_DIR / "outputs" / "14th_advanced_lightgbm_optuna"
STUDY_NAME = "f1_pitstop_advanced_lightgbm_14th"
STUDY_DB_PATH = DIAGNOSTIC_DIR / "optuna_study.db"

ID_COL = "id"
TARGET_COL = "PitNextLap"
HPO_TRIALS = 15
HPO_SPLITS = 3
HPO_SEED = 42
HPO_N_ESTIMATORS = 3500
HPO_EARLY_STOPPING = 150

FINAL_SPLITS = 5
FINAL_SEED = 42
FINAL_N_ESTIMATORS = 8000
FINAL_EARLY_STOPPING = 300

INNER_TE_SPLITS = 5
# The high-cardinality categorical features exceed the OpenCL GPU learner's
# bin-size limit on Kaggle, so this Optuna search intentionally runs on CPU.
USE_LIGHTGBM_GPU = False

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

RAW_CATEGORICAL_COLS = [
    "Driver",
    "Compound",
    "Race",
]

ENGINEERED_CATEGORICAL_COLS = [
    "Year_cat",
    "PitStop_cat",
    "LapNumber_cat",
    "Stint_cat",
    "TyreLife_cat",
    "Position_cat",
    "LapTime_cat",
    "LapTime_Delta_cat",
    "Cumulative_Degradation_cat",
    "RaceProgress_pct_cat",
    "Position_Change_cat",
    "LapNumberPerRaceProgress_cat",
    "TyreLifePerLapNumber_cat",
    "RaceProgress_bin_200",
    "LapTime_bin_7",
    "Race__Compound",
    "Race__Year",
]

TARGET_ENCODING_COLS = [
    "Race__Compound",
    "Race__Year",
]


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

    # This external-only column cannot be created reliably for competition test rows.
    external = external.drop(columns=["Normalized_TyreLife"], errors="ignore")

    return train, test, sample, external


def fit_quantile_edges(values, n_bins):
    numeric_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    edges = np.unique(
        np.quantile(
            numeric_values,
            np.linspace(0.0, 1.0, n_bins + 1),
        )
    )
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_quantile_bin(values, edges):
    return (
        pd.cut(
            pd.to_numeric(values, errors="coerce"),
            bins=edges,
            labels=False,
            include_lowest=True,
        )
        .fillna(-1)
        .astype("int32")
        .astype("string")
    )


def floor_as_string(values):
    return (
        np.floor(pd.to_numeric(values, errors="coerce"))
        .astype("Int64")
        .astype("string")
        .fillna("__MISSING__")
    )


def check_required_columns(frame, frame_name):
    required = set(RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS)
    missing_cols = sorted(required - set(frame.columns))
    if missing_cols:
        raise ValueError(f"{frame_name} is missing columns: {missing_cols}")


def make_base_features(raw, bin_edges):
    check_required_columns(raw, "Input data")
    features = pd.DataFrame(index=raw.index)

    for col in RAW_NUMERIC_COLS:
        features[col] = pd.to_numeric(raw[col], errors="coerce").astype("float32")

    lap_number = features["LapNumber"].clip(lower=1.0)
    race_progress = features["RaceProgress"].clip(lower=1e-6)
    degradation_abs = features["Cumulative_Degradation"].abs()

    # Arithmetic interactions adapted from the strong public RealMLP pipeline.
    features["LapNumberPerRaceProgress"] = features["LapNumber"] / race_progress
    features["TyreLifePerLapNumber"] = features["TyreLife"] / lap_number
    features["LapTimeXDegradation"] = (
        features["LapTime (s)"] * features["Cumulative_Degradation"]
    )
    features["LapTimeXAbsDegradation"] = features["LapTime (s)"] * degradation_abs
    features["LapTimePerAbsDegradation"] = (
        features["LapTime (s)"] / (degradation_abs + 1e-6)
    )

    for col in RAW_CATEGORICAL_COLS:
        features[col] = raw[col].astype("string").fillna("__MISSING__")

    features["Year_cat"] = floor_as_string(features["Year"])
    features["PitStop_cat"] = floor_as_string(features["PitStop"])
    features["LapNumber_cat"] = floor_as_string(features["LapNumber"])
    features["Stint_cat"] = floor_as_string(features["Stint"])
    features["TyreLife_cat"] = floor_as_string(features["TyreLife"])
    features["Position_cat"] = floor_as_string(features["Position"])
    features["LapTime_cat"] = floor_as_string(features["LapTime (s)"])
    features["LapTime_Delta_cat"] = floor_as_string(features["LapTime_Delta"])
    features["Cumulative_Degradation_cat"] = floor_as_string(
        features["Cumulative_Degradation"]
    )
    features["RaceProgress_pct_cat"] = floor_as_string(
        features["RaceProgress"] * 100.0
    )
    features["Position_Change_cat"] = floor_as_string(features["Position_Change"])
    features["LapNumberPerRaceProgress_cat"] = floor_as_string(
        features["LapNumberPerRaceProgress"]
    )
    features["TyreLifePerLapNumber_cat"] = floor_as_string(
        features["TyreLifePerLapNumber"]
    )
    features["RaceProgress_bin_200"] = apply_quantile_bin(
        features["RaceProgress"],
        bin_edges["RaceProgress"],
    )
    features["LapTime_bin_7"] = apply_quantile_bin(
        features["LapTime (s)"],
        bin_edges["LapTime (s)"],
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
        "Year_cat",
        "PitStop_cat",
        "Race__Compound",
        "Race__Year",
    ]

    for col in count_cols:
        count_map = combined[col].value_counts(dropna=False)
        count_name = f"{col}_count"
        train_features[count_name] = (
            train_features[col].map(count_map).fillna(0).astype("float32")
        )
        test_features[count_name] = (
            test_features[col].map(count_map).fillna(0).astype("float32")
        )
        external_features[count_name] = (
            external_features[col].map(count_map).fillna(0).astype("float32")
        )

    return train_features, test_features, external_features


def align_categories(train_features, test_features, external_features, categorical_cols):
    for col in categorical_cols:
        train_values = train_features[col].astype("string").fillna("__MISSING__")
        test_values = test_features[col].astype("string").fillna("__MISSING__")
        external_values = external_features[col].astype("string").fillna("__MISSING__")
        categories = pd.concat(
            [train_values, test_values, external_values],
            ignore_index=True,
        ).unique()

        train_features[col] = pd.Categorical(train_values, categories=categories)
        test_features[col] = pd.Categorical(test_values, categories=categories)
        external_features[col] = pd.Categorical(external_values, categories=categories)

    return train_features, test_features, external_features


def build_features(train, test, external):
    bin_edges = {
        "RaceProgress": fit_quantile_edges(train["RaceProgress"], n_bins=200),
        "LapTime (s)": fit_quantile_edges(train["LapTime (s)"], n_bins=7),
    }
    train_features = make_base_features(train, bin_edges)
    test_features = make_base_features(test, bin_edges)
    external_features = make_base_features(external, bin_edges)
    train_features, test_features, external_features = add_count_features(
        train_features,
        test_features,
        external_features,
    )

    categorical_cols = RAW_CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS
    train_features, test_features, external_features = align_categories(
        train_features,
        test_features,
        external_features,
        categorical_cols,
    )

    return train_features, test_features, external_features, categorical_cols


def weighted_target_encoding_map(keys, target, sample_weight, smoothing):
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
    encoding_map = (
        grouped["weighted_target_sum"] + smoothing * global_mean
    ) / (grouped["weight_sum"] + smoothing)

    return encoding_map, global_mean


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
    smoothing,
):
    train_features = train_features.copy()
    valid_features = valid_features.copy()
    if test_features is not None:
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
                smoothing,
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
            smoothing,
        )
        te_name = f"{col}_target_encoding"
        train_features[te_name] = encoded_train
        valid_features[te_name] = apply_target_encoding(
            valid_features[col],
            full_map,
            full_fallback,
        )
        if test_features is not None:
            test_features[te_name] = apply_target_encoding(
                test_features[col],
                full_map,
                full_fallback,
            )

    return train_features, valid_features, test_features


def base_params(seed, n_estimators):
    params = {
        "objective": "binary",
        "n_estimators": n_estimators,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.3,
        "reg_lambda": 3.0,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }

    return params


def suggest_params(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.018, 0.040, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 47, 63, 79, 95]),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            40,
            120,
            step=10,
        ),
        "subsample": trial.suggest_float("subsample", 0.72, 0.95),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 8.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.20),
        "external_weight": trial.suggest_float("external_weight", 0.45, 0.85, step=0.10),
        "te_smoothing": trial.suggest_categorical(
            "te_smoothing",
            [10.0, 20.0, 30.0, 50.0, 80.0],
        ),
    }


def split_training_params(tuned_params):
    model_params = tuned_params.copy()
    external_weight = model_params.pop("external_weight")
    te_smoothing = model_params.pop("te_smoothing")
    return model_params, external_weight, te_smoothing


def prepare_fold_data(
    X,
    y,
    X_external,
    y_external,
    train_idx,
    valid_idx,
    external_weight,
    te_smoothing,
    seed,
    X_test=None,
):
    X_train_comp = X.iloc[train_idx]
    y_train_comp = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    X_train = pd.concat([X_train_comp, X_external], ignore_index=True)
    y_train = pd.concat([y_train_comp, y_external], ignore_index=True)
    sample_weight = np.concatenate(
        [
            np.ones(len(X_train_comp), dtype=float),
            np.full(len(X_external), external_weight, dtype=float),
        ]
    )
    X_train, X_valid, X_fold_test = add_fold_target_encoding(
        X_train,
        y_train,
        sample_weight,
        X_valid,
        X_test,
        seed=seed,
        smoothing=te_smoothing,
    )
    return X_train, y_train, sample_weight, X_valid, y_valid, X_fold_test


def fit_model(
    X_train,
    y_train,
    sample_weight,
    X_valid,
    y_valid,
    categorical_cols,
    params,
    early_stopping_rounds,
    log_period,
):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            ),
            lgb.log_evaluation(period=log_period),
        ],
    )
    return model


def run_optuna_search(X, y, X_external, y_external, categorical_cols):
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{STUDY_DB_PATH.resolve().as_posix()}"
    folds = list(
        StratifiedKFold(
            n_splits=HPO_SPLITS,
            shuffle=True,
            random_state=HPO_SEED,
        ).split(X, y)
    )

    def objective(trial):
        tuned_params = suggest_params(trial)
        model_params, external_weight, te_smoothing = split_training_params(
            tuned_params
        )
        fold_aucs = []

        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            (
                X_train,
                y_train,
                sample_weight,
                X_valid,
                y_valid,
                _,
            ) = prepare_fold_data(
                X,
                y,
                X_external,
                y_external,
                train_idx,
                valid_idx,
                external_weight,
                te_smoothing,
                seed=HPO_SEED + fold,
            )
            params = base_params(HPO_SEED + fold, HPO_N_ESTIMATORS)
            params.update(model_params)
            model = fit_model(
                X_train,
                y_train,
                sample_weight,
                X_valid,
                y_valid,
                categorical_cols,
                params,
                early_stopping_rounds=HPO_EARLY_STOPPING,
                log_period=0,
            )
            valid_prediction = model.predict_proba(
                X_valid,
                num_iteration=model.best_iteration_,
            )[:, 1]
            fold_aucs.append(roc_auc_score(y_valid, valid_prediction))
            trial.report(float(np.mean(fold_aucs)), step=fold)

            del model, X_train, X_valid
            gc.collect()

            if fold >= 2 and trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_aucs", fold_aucs)
        return float(np.mean(fold_aucs))

    def print_trial_result(study, trial):
        study.trials_dataframe().to_csv(
            DIAGNOSTIC_DIR / "optuna_trials.csv",
            index=False,
        )
        if trial.value is None:
            print(f"Trial {trial.number + 1:>2}/{HPO_TRIALS}: pruned")
        else:
            print(
                f"Trial {trial.number + 1:>2}/{HPO_TRIALS}: "
                f"AUC={trial.value:.6f} | Best={study.best_value:.6f}"
            )

    print("\n" + "=" * 60)
    print("Advanced LightGBM Optuna search")
    print("=" * 60)
    print(f"Trials: {HPO_TRIALS}")
    print(f"CV: {HPO_SPLITS}-fold")
    print(f"Try LightGBM GPU: {USE_LIGHTGBM_GPU}")

    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        sampler=optuna.samplers.TPESampler(seed=HPO_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    remaining_trials = max(0, HPO_TRIALS - len(study.trials))
    print(f"Saved trials: {len(study.trials)}")
    print(f"Remaining trials: {remaining_trials}")

    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[print_trial_result],
        )
    else:
        print("Optuna search is already complete. Reusing the saved study.")

    print("\nBest HPO AUC:")
    print(f"{study.best_value:.6f}")
    print("\nBest parameters:")
    print(json.dumps(study.best_params, indent=2))
    return study


def collect_feature_importance(model, feature_names, fold):
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    gain_total = gain.sum()
    split_total = split.sum()

    return pd.DataFrame(
        {
            "feature": feature_names,
            "gain_percent": gain / gain_total * 100 if gain_total else 0.0,
            "split_percent": split / split_total * 100 if split_total else 0.0,
            "fold": fold,
        }
    )


def train_final_model(
    X,
    y,
    X_test,
    X_external,
    y_external,
    categorical_cols,
    tuned_params,
):
    model_params, external_weight, te_smoothing = split_training_params(tuned_params)
    folds = StratifiedKFold(
        n_splits=FINAL_SPLITS,
        shuffle=True,
        random_state=FINAL_SEED,
    )
    oof_prediction = np.zeros(len(X), dtype=float)
    test_prediction = np.zeros(len(X_test), dtype=float)
    fold_records = []
    importance_frames = []

    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        print("\n" + "=" * 60)
        print(f"Final Fold {fold}/{FINAL_SPLITS}")
        print("=" * 60)
        (
            X_train,
            y_train,
            sample_weight,
            X_valid,
            y_valid,
            X_fold_test,
        ) = prepare_fold_data(
            X,
            y,
            X_external,
            y_external,
            train_idx,
            valid_idx,
            external_weight,
            te_smoothing,
            seed=FINAL_SEED + fold,
            X_test=X_test,
        )
        params = base_params(FINAL_SEED + fold, FINAL_N_ESTIMATORS)
        params.update(model_params)
        model = fit_model(
            X_train,
            y_train,
            sample_weight,
            X_valid,
            y_valid,
            categorical_cols,
            params,
            early_stopping_rounds=FINAL_EARLY_STOPPING,
            log_period=500,
        )
        valid_prediction = model.predict_proba(
            X_valid,
            num_iteration=model.best_iteration_,
        )[:, 1]
        fold_test_prediction = model.predict_proba(
            X_fold_test,
            num_iteration=model.best_iteration_,
        )[:, 1]
        oof_prediction[valid_idx] = valid_prediction
        test_prediction += fold_test_prediction / FINAL_SPLITS

        fold_auc = roc_auc_score(y_valid, valid_prediction)
        fold_records.append(
            {
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": model.best_iteration_,
                "competition_train_rows": len(train_idx),
                "external_train_rows": len(X_external),
                "external_weight": external_weight,
                "te_smoothing": te_smoothing,
            }
        )
        importance_frames.append(
            collect_feature_importance(model, X_train.columns, fold)
        )
        print(
            f"Final Fold {fold}: AUC={fold_auc:.6f}, "
            f"best_iteration={model.best_iteration_}"
        )

        del model, X_train, X_valid, X_fold_test
        gc.collect()

    ensemble_auc = roc_auc_score(y, oof_prediction)
    fold_metrics = pd.DataFrame(fold_records)
    importance_detail = pd.concat(importance_frames, ignore_index=True)
    importance_summary = (
        importance_detail.groupby("feature", as_index=False)
        .agg(
            gain_mean=("gain_percent", "mean"),
            gain_std=("gain_percent", "std"),
            split_mean=("split_percent", "mean"),
        )
        .sort_values("gain_mean", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("Pre-submission diagnostics")
    print("=" * 60)
    print(f"Optuna-tuned advanced LightGBM OOF AUC: {ensemble_auc:.6f}")
    print(
        "Fold AUC: "
        f"mean={fold_metrics['auc'].mean():.6f}, "
        f"std={fold_metrics['auc'].std():.6f}, "
        f"min={fold_metrics['auc'].min():.6f}, "
        f"max={fold_metrics['auc'].max():.6f}"
    )
    print("\nTest prediction distribution:")
    print(pd.Series(test_prediction, name=TARGET_COL).describe().to_string())
    print("\nTop feature importance by gain (%):")
    print(importance_summary.head(25).to_string(index=False))

    return {
        "oof_prediction": np.clip(oof_prediction, 0.0, 1.0),
        "test_prediction": np.clip(test_prediction, 0.0, 1.0),
        "ensemble_auc": ensemble_auc,
        "fold_metrics": fold_metrics,
        "importance_summary": importance_summary,
        "external_weight": external_weight,
        "te_smoothing": te_smoothing,
    }


def save_diagnostics(train, external, result, study):
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET_COL: train[TARGET_COL],
            "oof_prediction": result["oof_prediction"],
        }
    ).to_csv(DIAGNOSTIC_DIR / "oof_predictions.csv", index=False)
    result["fold_metrics"].to_csv(DIAGNOSTIC_DIR / "fold_metrics.csv", index=False)
    result["importance_summary"].to_csv(
        DIAGNOSTIC_DIR / "feature_importance.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "metric": [
                "advanced_lightgbm_oof_auc",
                "hpo_best_auc",
                "competition_train_rows",
                "external_train_rows",
                "external_weight",
                "te_smoothing",
            ],
            "value": [
                result["ensemble_auc"],
                study.best_value,
                len(train),
                len(external),
                result["external_weight"],
                result["te_smoothing"],
            ],
        }
    ).to_csv(DIAGNOSTIC_DIR / "run_summary.csv", index=False)
    study.trials_dataframe().to_csv(
        DIAGNOSTIC_DIR / "optuna_trials.csv",
        index=False,
    )
    with (DIAGNOSTIC_DIR / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)
    print(f"\nSaved diagnostics: {DIAGNOSTIC_DIR}")


def make_submission(sample, test, prediction):
    submission = sample.copy()
    if ID_COL in test.columns:
        submission[ID_COL] = test[ID_COL].values
    submission[TARGET_COL] = prediction
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved submission: {OUTPUT_PATH}")
    print(submission.head())


def main():
    train, test, sample, external = load_data()
    X, X_test, X_external, categorical_cols = build_features(
        train,
        test,
        external,
    )
    y = train[TARGET_COL].astype(int).reset_index(drop=True)
    y_external = external[TARGET_COL].astype(int).reset_index(drop=True)

    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {WORK_DIR}")
    print(f"Competition train shape: {train.shape}")
    print(f"External train shape: {external.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Base features before fold TE: {len(X.columns)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Fold-safe target encoding features: {TARGET_ENCODING_COLS}")
    print(f"Try LightGBM GPU: {USE_LIGHTGBM_GPU}")

    study = run_optuna_search(
        X,
        y,
        X_external,
        y_external,
        categorical_cols,
    )
    result = train_final_model(
        X,
        y,
        X_test,
        X_external,
        y_external,
        categorical_cols,
        study.best_params,
    )
    save_diagnostics(train, external, result, study)
    make_submission(sample, test, result["test_prediction"])


if __name__ == "__main__":
    main()
