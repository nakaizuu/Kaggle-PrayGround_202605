from pathlib import Path
import warnings

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Required package is missing. Install dependencies with:\n"
        "pip install lightgbm scikit-learn pandas numpy"
    ) from exc


warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = DATA_DIR / "submission_lightgbm_lite_fe.csv"

ID_COL = "id"
TARGET_COL = "PitNextLap"
RANDOM_STATE = 42
N_SPLITS = 5
SUBMIT_PROBABILITY = True


def safe_divide(a, b):
    return a / np.where(np.abs(b) < 1e-9, np.nan, b)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in train.csv")
    if ID_COL not in sample.columns:
        raise ValueError(f"ID column '{ID_COL}' was not found in sample_submission.csv")

    return train, test, sample


def add_lite_features(df):
    df = df.copy()

    lap_number = df["LapNumber"].astype(float)
    stint = df["Stint"].astype(float)
    tyre_life = df["TyreLife"].astype(float)
    race_progress = df["RaceProgress"].astype(float)
    position = df["Position"].astype(float)
    position_change = df["Position_Change"].astype(float)
    lap_time = df["LapTime (s)"].astype(float)
    lap_time_delta = df["LapTime_Delta"].astype(float)
    cumulative_degradation = df["Cumulative_Degradation"].astype(float)

    estimated_total_laps = safe_divide(lap_number, race_progress)
    laps_remaining = estimated_total_laps - lap_number

    df["EstimatedTotalLaps"] = estimated_total_laps
    df["LapsRemaining"] = laps_remaining
    df["RaceProgressRemaining"] = 1.0 - race_progress
    df["TyreLifeRatio"] = safe_divide(tyre_life, estimated_total_laps)
    df["TyreLifePerLap"] = safe_divide(tyre_life, lap_number)
    df["TyreLifePerStint"] = safe_divide(tyre_life, stint)

    df["DegradationPerTyreLife"] = safe_divide(cumulative_degradation, tyre_life + 1.0)
    df["DegradationPerLap"] = safe_divide(cumulative_degradation, lap_number)
    df["LapTimeDeltaAbs"] = lap_time_delta.abs()
    df["LapTimeVsDeltaRatio"] = safe_divide(lap_time_delta, lap_time)

    df["PositionAfterChange"] = position + position_change
    df["AbsPositionChange"] = position_change.abs()
    df["PositionGainFlag"] = (position_change < 0).astype(int)
    df["PositionLossFlag"] = (position_change > 0).astype(int)
    df["FrontRunnerFlag"] = (position <= 5).astype(int)

    compound_hardness = {
        "SOFT": 1,
        "MEDIUM": 2,
        "HARD": 3,
        "INTERMEDIATE": 0,
        "WET": -1,
    }
    df["CompoundHardness"] = df["Compound"].map(compound_hardness).fillna(0).astype(float)
    df["TyreLifeXHardness"] = tyre_life * df["CompoundHardness"]
    df["RemainingXHardness"] = laps_remaining * df["CompoundHardness"]

    df["DriverRace"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
    df["RaceCompound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
    df["DriverCompound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)

    return df.replace([np.inf, -np.inf], np.nan)


def build_features(train, test):
    train_fe = add_lite_features(train)
    test_fe = add_lite_features(test)

    drop_cols = {TARGET_COL, ID_COL}
    feature_cols = [
        col for col in train_fe.columns if col not in drop_cols and col in test_fe.columns
    ]

    X = train_fe[feature_cols].copy()
    X_test = test_fe[feature_cols].copy()

    categorical_cols = [
        col
        for col in feature_cols
        if (
            not pd.api.types.is_numeric_dtype(X[col])
            or not pd.api.types.is_numeric_dtype(X_test[col])
        )
    ]

    for col in categorical_cols:
        train_values = X[col].astype("string").fillna("__MISSING__")
        test_values = X_test[col].astype("string").fillna("__MISSING__")
        categories = pd.concat([train_values, test_values], ignore_index=True).unique()

        X[col] = pd.Categorical(train_values, categories=categories)
        X_test[col] = pd.Categorical(test_values, categories=categories)

    return X, X_test, categorical_cols


def train_and_predict(X, y, X_test, categorical_cols):
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    params = {
        "objective": "binary",
        "n_estimators": 5000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        valid_pred = model.predict_proba(
            X_valid, num_iteration=model.best_iteration_
        )[:, 1]
        fold_test_pred = model.predict_proba(
            X_test, num_iteration=model.best_iteration_
        )[:, 1]

        oof_pred[valid_idx] = valid_pred
        test_pred += fold_test_pred / N_SPLITS

        fold_auc = roc_auc_score(y_valid, valid_pred)
        print(f"Fold {fold}: AUC = {fold_auc:.6f}, best_iteration = {model.best_iteration_}")

    overall_auc = roc_auc_score(y, oof_pred)
    print(f"OOF AUC: {overall_auc:.6f}")

    return np.clip(test_pred, 0.0, 1.0)


def make_submission(sample, test, pred):
    target_candidates = [col for col in sample.columns if col != ID_COL]
    if TARGET_COL in sample.columns:
        submit_col = TARGET_COL
    elif len(target_candidates) == 1:
        submit_col = target_candidates[0]
    else:
        raise ValueError("Could not determine submission target column.")

    submission = sample.copy()
    if ID_COL in test.columns:
        submission[ID_COL] = test[ID_COL].values

    submission[submit_col] = pred if SUBMIT_PROBABILITY else (pred >= 0.5).astype(int)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(submission.head())


def main():
    train, test, sample = load_data()
    X, X_test, categorical_cols = build_features(train, test)
    y = train[TARGET_COL].astype(int)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Features after lite FE: {len(X.columns)}")
    print(f"Categorical features: {categorical_cols}")

    pred = train_and_predict(X, y, X_test, categorical_cols)
    make_submission(sample, test, pred)


if __name__ == "__main__":
    main()
