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
OUTPUT_PATH = DATA_DIR / "submission_lightgbm_tuned_seed_ensemble.csv"
DIAGNOSTIC_DIR = DATA_DIR / "outputs" / "5th_tuned_seed_ensemble"

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
        "n_estimators": 8000,  # 4th: 5000. Allow more trees for the lower learning rate.
        "learning_rate": 0.025,  # 4th: 0.03. Learn a little more carefully.
        "num_leaves": 63,
        "max_depth": 8,  # 4th: -1. Limit tree depth to reduce overfitting.
        "min_child_samples": 60,  # 4th: 40. Avoid very small leaf groups.
        "subsample": 0.85,
        "subsample_freq": 1,  # 4th: unset (= 0). Actually enable row subsampling.
        "colsample_bytree": 0.80,  # 4th: 0.85. Add slightly more feature randomness.
        "reg_alpha": 0.3,  # 4th: 0.1. Stronger L1 regularization.
        "reg_lambda": 3.0,  # 4th: 1.0. Stronger L2 regularization.
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def collect_feature_importance(model, feature_names, seed, fold):
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")

    gain_total = gain.sum()
    split_total = split.sum()

    return pd.DataFrame(
        {
            "feature": feature_names,
            "gain_percent": gain / gain_total * 100 if gain_total else 0.0,
            "split_percent": split / split_total * 100 if split_total else 0.0,
            "seed": seed,
            "fold": fold,
        }
    )


def train_one_seed(X, y, X_test, categorical_cols, seed):
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_records = []
    importance_frames = []

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
                lgb.early_stopping(
                    stopping_rounds=300,  # 4th: 200. Wait longer for slower learning.
                    verbose=False,
                ),
                lgb.log_evaluation(period=500),
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
        fold_records.append(
            {
                "seed": seed,
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": model.best_iteration_,
            }
        )
        importance_frames.append(
            collect_feature_importance(model, X.columns, seed, fold)
        )

        print(
            f"Seed {seed} Fold {fold}: "
            f"AUC = {fold_auc:.6f}, best_iteration = {model.best_iteration_}"
        )

    seed_auc = roc_auc_score(y, oof_pred)
    print(f"Seed {seed} OOF AUC: {seed_auc:.6f}")

    return (
        oof_pred,
        test_pred,
        fold_records,
        pd.concat(importance_frames, ignore_index=True),
        {"seed": seed, "oof_auc": seed_auc},
    )


def train_and_predict(X, y, X_test, categorical_cols):
    all_oof = []
    all_test = []
    all_fold_records = []
    all_importance = []
    seed_records = []

    for seed in SEEDS:
        oof_pred, test_pred, fold_records, importance, seed_record = train_one_seed(
            X, y, X_test, categorical_cols, seed
        )
        all_oof.append(oof_pred)
        all_test.append(test_pred)
        all_fold_records.extend(fold_records)
        all_importance.append(importance)
        seed_records.append(seed_record)

    ensemble_oof = np.mean(all_oof, axis=0)
    ensemble_test = np.mean(all_test, axis=0)
    ensemble_auc = roc_auc_score(y, ensemble_oof)

    fold_metrics = pd.DataFrame(all_fold_records)
    seed_metrics = pd.DataFrame(seed_records)
    importance_detail = pd.concat(all_importance, ignore_index=True)
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
    print(f"Ensemble OOF AUC: {ensemble_auc:.6f}")
    print(
        "Fold AUC: "
        f"mean={fold_metrics['auc'].mean():.6f}, "
        f"std={fold_metrics['auc'].std():.6f}, "
        f"min={fold_metrics['auc'].min():.6f}, "
        f"max={fold_metrics['auc'].max():.6f}"
    )
    print("\nSeed OOF AUC:")
    print(seed_metrics.to_string(index=False))
    print("\nTest prediction distribution:")
    print(pd.Series(ensemble_test, name=TARGET_COL).describe().to_string())
    print("\nTop feature importance by gain (%):")
    print(importance_summary.head(15).to_string(index=False))

    return {
        "test_pred": np.clip(ensemble_test, 0.0, 1.0),
        "oof_pred": np.clip(ensemble_oof, 0.0, 1.0),
        "ensemble_auc": ensemble_auc,
        "fold_metrics": fold_metrics,
        "seed_metrics": seed_metrics,
        "importance_summary": importance_summary,
    }


def save_diagnostics(train, result):
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET_COL: train[TARGET_COL],
            "oof_prediction": result["oof_pred"],
        }
    ).to_csv(DIAGNOSTIC_DIR / "oof_predictions.csv", index=False)

    result["fold_metrics"].to_csv(DIAGNOSTIC_DIR / "fold_metrics.csv", index=False)
    result["seed_metrics"].to_csv(DIAGNOSTIC_DIR / "seed_metrics.csv", index=False)
    result["importance_summary"].to_csv(
        DIAGNOSTIC_DIR / "feature_importance.csv", index=False
    )

    pd.DataFrame(
        {
            "metric": ["ensemble_oof_auc"],
            "value": [result["ensemble_auc"]],
        }
    ).to_csv(DIAGNOSTIC_DIR / "run_summary.csv", index=False)

    print(f"\nSaved diagnostics: {DIAGNOSTIC_DIR}")


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
    print(f"Saved submission: {OUTPUT_PATH}")
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
    print(f"Positive rate: {y.mean():.6f}")

    result = train_and_predict(X, y, X_test, categorical_cols)
    save_diagnostics(train, result)
    make_submission(sample, test, result["test_pred"])


if __name__ == "__main__":
    main()
