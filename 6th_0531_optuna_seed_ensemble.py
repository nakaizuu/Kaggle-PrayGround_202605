from pathlib import Path
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

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = DATA_DIR / "submission_lightgbm_optuna_seed_ensemble.csv"
DIAGNOSTIC_DIR = DATA_DIR / "outputs" / "6th_optuna_seed_ensemble"

ID_COL = "id"
TARGET_COL = "PitNextLap"

# Search with fewer folds, then train the final ensemble carefully.
HPO_TRIALS = 20
HPO_SPLITS = 3
HPO_SEED = 42
HPO_N_ESTIMATORS = 3500
HPO_EARLY_STOPPING = 150

FINAL_SPLITS = 5
FINAL_SEEDS = [42, 2025, 3407]
FINAL_N_ESTIMATORS = 8000
FINAL_EARLY_STOPPING = 300

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


def fifth_params(seed):
    return {
        "objective": "binary",
        "n_estimators": FINAL_N_ESTIMATORS,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.3,
        "reg_lambda": 3.0,
        "min_split_gain": 0.0,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }


def suggest_params(trial):
    # Search near the successful 5th settings instead of exploring a huge space.
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.018, 0.040, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 47, 63, 79, 95, 127]),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 120, step=10),
        "subsample": trial.suggest_float("subsample", 0.72, 0.95),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 6.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.20),
    }


def run_optuna_search(X, y, categorical_cols):
    folds = list(
        StratifiedKFold(
            n_splits=HPO_SPLITS,
            shuffle=True,
            random_state=HPO_SEED,
        ).split(X, y)
    )

    def objective(trial):
        tuned_params = suggest_params(trial)
        fold_aucs = []

        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            params = fifth_params(HPO_SEED + fold)
            params.update(tuned_params)
            params["n_estimators"] = HPO_N_ESTIMATORS

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="auc",
                categorical_feature=categorical_cols,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=HPO_EARLY_STOPPING,
                        verbose=False,
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )

            valid_pred = model.predict_proba(
                X_valid, num_iteration=model.best_iteration_
            )[:, 1]
            fold_aucs.append(roc_auc_score(y_valid, valid_pred))

        mean_auc = float(np.mean(fold_aucs))
        trial.set_user_attr("fold_aucs", fold_aucs)
        return mean_auc

    def print_trial_result(study, trial):
        print(
            f"Trial {trial.number + 1:>2}/{HPO_TRIALS}: "
            f"AUC = {trial.value:.6f} | "
            f"Best = {study.best_value:.6f}"
        )

    print("\n" + "=" * 60)
    print("Optuna hyperparameter search")
    print("=" * 60)
    print(f"Trials: {HPO_TRIALS}")
    print(f"CV: {HPO_SPLITS}-fold")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=HPO_SEED),
        study_name="f1_pitstop_lgbm_6th",
    )
    study.optimize(
        objective,
        n_trials=HPO_TRIALS,
        callbacks=[print_trial_result],
    )

    print("\nBest HPO AUC:")
    print(f"{study.best_value:.6f}")
    print("\nBest parameters:")
    print(json.dumps(study.best_params, indent=2))

    return study


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


def train_one_seed(X, y, X_test, categorical_cols, seed, tuned_params):
    folds = StratifiedKFold(n_splits=FINAL_SPLITS, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_records = []
    importance_frames = []

    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        params = fifth_params(seed + fold)
        params.update(tuned_params)
        params["n_estimators"] = FINAL_N_ESTIMATORS

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=FINAL_EARLY_STOPPING,
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
        test_pred += fold_test_pred / FINAL_SPLITS

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


def train_final_ensemble(X, y, X_test, categorical_cols, tuned_params):
    all_oof = []
    all_test = []
    all_fold_records = []
    all_importance = []
    seed_records = []

    print("\n" + "=" * 60)
    print("Final 5-fold x 3-seed ensemble")
    print("=" * 60)

    for seed in FINAL_SEEDS:
        oof_pred, test_pred, fold_records, importance, seed_record = train_one_seed(
            X, y, X_test, categorical_cols, seed, tuned_params
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


def save_diagnostics(train, study, result):
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(study.trials_dataframe()).to_csv(
        DIAGNOSTIC_DIR / "hpo_trials.csv", index=False
    )
    with (DIAGNOSTIC_DIR / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)

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
            "metric": ["hpo_best_auc", "ensemble_oof_auc"],
            "value": [study.best_value, result["ensemble_auc"]],
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
    print(f"Positive rate: {y.mean():.6f}")

    study = run_optuna_search(X, y, categorical_cols)
    result = train_final_ensemble(X, y, X_test, categorical_cols, study.best_params)
    save_diagnostics(train, study, result)
    make_submission(sample, test, result["test_pred"])


if __name__ == "__main__":
    main()
