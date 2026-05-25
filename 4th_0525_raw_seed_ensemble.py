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
OUTPUT_PATH = DATA_DIR / "submission_lightgbm_raw_seed_ensemble.csv"

ID_COL = "id"
TARGET_COL = "PitNextLap"
N_SPLITS = 5
SEEDS = [42, 2025, 3407]
SUBMIT_PROBABILITY = True


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in train.csv")
    if ID_COL not in sample.columns:
        raise ValueError(f"ID column '{ID_COL}' was not found in sample_submission.csv")

    return train, test, sample


def build_features(train, test):
    drop_cols = {TARGET_COL, ID_COL}
    feature_cols = [col for col in train.columns if col not in drop_cols and col in test.columns]

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

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


def base_params(seed):
    return {
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
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def train_one_seed(X, y, X_test, categorical_cols, seed):
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**base_params(seed + fold))
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
        print(
            f"Seed {seed} Fold {fold}: "
            f"AUC = {fold_auc:.6f}, best_iteration = {model.best_iteration_}"
        )

    seed_auc = roc_auc_score(y, oof_pred)
    print(f"Seed {seed} OOF AUC: {seed_auc:.6f}")

    return oof_pred, test_pred


def train_and_predict(X, y, X_test, categorical_cols):
    all_oof = []
    all_test = []

    for seed in SEEDS:
        oof_pred, test_pred = train_one_seed(X, y, X_test, categorical_cols, seed)
        all_oof.append(oof_pred)
        all_test.append(test_pred)

    ensemble_oof = np.mean(all_oof, axis=0)
    ensemble_test = np.mean(all_test, axis=0)

    ensemble_auc = roc_auc_score(y, ensemble_oof)
    print(f"Ensemble OOF AUC: {ensemble_auc:.6f}")

    return np.clip(ensemble_test, 0.0, 1.0)


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
    print(f"Features: {len(X.columns)}")
    print(f"Categorical features: {categorical_cols}")
    print(f"Seeds: {SEEDS}")

    pred = train_and_predict(X, y, X_test, categorical_cols)
    make_submission(sample, test, pred)


if __name__ == "__main__":
    main()
