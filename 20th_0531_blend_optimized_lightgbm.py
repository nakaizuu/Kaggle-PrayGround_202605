from pathlib import Path
import gc
import json
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
EXTERNAL_PATH = DATA_DIR / "f1_strategy_dataset_v4.csv"
DIAGNOSTIC_DIR = DATA_DIR / "outputs" / "20th_blend_optimized_lightgbm"
STANDALONE_OUTPUT_PATH = (
    DATA_DIR / "submission_20th_blend_optimized_lightgbm.csv"
)
BLEND_OUTPUT_PATH = (
    DATA_DIR / "submission_20th_realmlp_lightgbm_probability_blend.csv"
)

ID_COL = "id"
TARGET_COL = "PitNextLap"
OOF_PRED_COL = "oof_prediction"
SEARCH_SPLITS = 3
SEARCH_SEED = 42
FINAL_SPLITS = 5
FINAL_SEEDS = [42, 2025, 3407]
BLEND_WEIGHTS = np.arange(0.00, 0.301, 0.01)

# Prefer 18th after its Kaggle output has been copied locally. Until then,
# 17th remains a useful anchor for running this CPU experiment in parallel.
ANCHOR_SPECS = [
    {
        "name": "realmlp_18",
        "label": "18th GPU RealMLP seed ensemble",
        "submission": DATA_DIR / "submission_18th_gpu_realmlp_seed_ensemble.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "18th_gpu_realmlp_seed_ensemble"
            / "oof_predictions.csv",
        ],
    },
    {
        "name": "realmlp_17",
        "label": "17th GPU RealMLP",
        "submission": DATA_DIR / "submission_17th_gpu_realmlp_reference.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "17th_gpu_realmlp_reference"
            / "oof_predictions.csv",
            DATA_DIR / "outputs" / "17th_super_pytorch" / "oof_predictions.csv",
        ],
    },
]

FEATURE_SETS = {
    "raw_all": None,
    "raw_no_driver": {
        "exclude": ["Driver"],
    },
    "strategy_core": {
        "include": [
            "Year",
            "PitStop",
            "LapNumber",
            "Stint",
            "TyreLife",
            "Position",
            "RaceProgress",
            "Compound",
            "Race",
        ],
    },
}

MODEL_PROFILES = {
    "baseline": {},
    "shallow_diverse": {
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 100,
        "subsample": 0.75,
        "subsample_freq": 2,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.6,
        "reg_lambda": 4.5,
    },
    "sparse_diverse": {
        "num_leaves": 47,
        "max_depth": 7,
        "min_child_samples": 90,
        "subsample": 0.78,
        "subsample_freq": 3,
        "colsample_bytree": 0.62,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
}

# This is deliberately small enough for a local CPU run. The goal is not to
# maximize standalone LightGBM AUC, but to find a useful RealMLP complement.
SEARCH_VARIANTS = [
    {
        "name": "baseline_all_ext45",
        "feature_set": "raw_all",
        "profile": "baseline",
        "external_weight": 0.45,
    },
    {
        "name": "baseline_all_ext65",
        "feature_set": "raw_all",
        "profile": "baseline",
        "external_weight": 0.65,
    },
    {
        "name": "baseline_all_ext85",
        "feature_set": "raw_all",
        "profile": "baseline",
        "external_weight": 0.85,
    },
    {
        "name": "no_driver_ext45",
        "feature_set": "raw_no_driver",
        "profile": "baseline",
        "external_weight": 0.45,
    },
    {
        "name": "no_driver_ext65",
        "feature_set": "raw_no_driver",
        "profile": "baseline",
        "external_weight": 0.65,
    },
    {
        "name": "no_driver_ext85",
        "feature_set": "raw_no_driver",
        "profile": "baseline",
        "external_weight": 0.85,
    },
    {
        "name": "strategy_core_ext45",
        "feature_set": "strategy_core",
        "profile": "baseline",
        "external_weight": 0.45,
    },
    {
        "name": "strategy_core_ext65",
        "feature_set": "strategy_core",
        "profile": "baseline",
        "external_weight": 0.65,
    },
    {
        "name": "strategy_core_ext85",
        "feature_set": "strategy_core",
        "profile": "baseline",
        "external_weight": 0.85,
    },
    {
        "name": "shallow_all_ext65",
        "feature_set": "raw_all",
        "profile": "shallow_diverse",
        "external_weight": 0.65,
    },
    {
        "name": "shallow_no_driver_ext65",
        "feature_set": "raw_no_driver",
        "profile": "shallow_diverse",
        "external_weight": 0.65,
    },
    {
        "name": "sparse_all_ext65",
        "feature_set": "raw_all",
        "profile": "sparse_diverse",
        "external_weight": 0.65,
    },
]


def find_existing_path(candidates):
    return next((path for path in candidates if path.exists()), None)


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

    external = external.drop(columns=["Normalized_TyreLife"], errors="ignore")
    return train, test, sample, external


def load_anchor(train, test):
    for spec in ANCHOR_SPECS:
        oof_path = find_existing_path(spec["oof_candidates"])
        submission_path = spec["submission"]
        if oof_path is None or not submission_path.exists():
            continue

        oof = pd.read_csv(oof_path)
        submission = pd.read_csv(submission_path)
        required_oof = {ID_COL, TARGET_COL, OOF_PRED_COL}
        required_submission = {ID_COL, TARGET_COL}
        if not required_oof.issubset(oof.columns):
            raise ValueError(f"{spec['name']} OOF is missing required columns.")
        if not required_submission.issubset(submission.columns):
            raise ValueError(f"{spec['name']} submission is missing required columns.")

        aligned_oof = train[[ID_COL, TARGET_COL]].merge(
            oof[[ID_COL, TARGET_COL, OOF_PRED_COL]],
            on=ID_COL,
            how="left",
            suffixes=("_train", "_anchor"),
            validate="one_to_one",
        )
        aligned_test = test[[ID_COL]].merge(
            submission[[ID_COL, TARGET_COL]],
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
        if aligned_oof[OOF_PRED_COL].isna().any():
            raise ValueError(f"{spec['name']} OOF could not be aligned by ID.")
        if aligned_test[TARGET_COL].isna().any():
            raise ValueError(f"{spec['name']} submission could not be aligned by ID.")
        if not np.allclose(
            aligned_oof[f"{TARGET_COL}_train"],
            aligned_oof[f"{TARGET_COL}_anchor"],
        ):
            raise ValueError(f"{spec['name']} OOF target values do not match train.csv.")

        print(
            f"Using anchor: {spec['name']} ({spec['label']})\n"
            f"  OOF: {oof_path.relative_to(DATA_DIR)}\n"
            f"  submission: {submission_path.name}"
        )
        return {
            "name": spec["name"],
            "label": spec["label"],
            "oof_prediction": aligned_oof[OOF_PRED_COL].to_numpy(dtype=float),
            "test_prediction": aligned_test[TARGET_COL].to_numpy(dtype=float),
        }

    raise FileNotFoundError(
        "No RealMLP anchor was found. Copy the 18th or 17th submission and "
        "OOF predictions into the local project directory."
    )


def choose_feature_cols(train, test, feature_set_name):
    drop_cols = {TARGET_COL, ID_COL}
    available_cols = [
        col
        for col in train.columns
        if col not in drop_cols and col in test.columns
    ]
    spec = FEATURE_SETS[feature_set_name]
    if spec is None:
        return available_cols
    if "include" in spec:
        missing = sorted(set(spec["include"]) - set(available_cols))
        if missing:
            raise ValueError(f"{feature_set_name} is missing columns: {missing}")
        return spec["include"]
    return [col for col in available_cols if col not in set(spec["exclude"])]


def build_features(train, test, external, feature_set_name):
    feature_cols = choose_feature_cols(train, test, feature_set_name)
    missing_external_cols = [
        col for col in feature_cols if col not in external.columns
    ]
    if missing_external_cols:
        raise ValueError(f"External data is missing columns: {missing_external_cols}")

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    X_external = external[feature_cols].copy()
    categorical_cols = [
        col
        for col in feature_cols
        if (
            not pd.api.types.is_numeric_dtype(X[col])
            or not pd.api.types.is_numeric_dtype(X_test[col])
            or not pd.api.types.is_numeric_dtype(X_external[col])
        )
    ]

    for col in categorical_cols:
        train_values = X[col].astype("string").fillna("__MISSING__")
        test_values = X_test[col].astype("string").fillna("__MISSING__")
        external_values = X_external[col].astype("string").fillna("__MISSING__")
        categories = pd.concat(
            [train_values, test_values, external_values],
            ignore_index=True,
        ).unique()
        X[col] = pd.Categorical(train_values, categories=categories)
        X_test[col] = pd.Categorical(test_values, categories=categories)
        X_external[col] = pd.Categorical(external_values, categories=categories)

    return X, X_test, X_external, categorical_cols


def base_params(seed, profile_name):
    params = {
        "objective": "binary",
        "n_estimators": 8000,
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
    params.update(MODEL_PROFILES[profile_name])
    return params


def evaluate_blend(y, anchor_prediction, support_prediction):
    records = []
    for support_weight in BLEND_WEIGHTS:
        blended = (
            (1.0 - support_weight) * anchor_prediction
            + support_weight * support_prediction
        )
        records.append(
            {
                "realmlp_weight": 1.0 - support_weight,
                "lightgbm_weight": support_weight,
                "blend_auc": roc_auc_score(y, blended),
            }
        )
    metrics = pd.DataFrame(records)
    best = metrics.sort_values(
        ["blend_auc", "realmlp_weight"],
        ascending=[False, False],
    ).iloc[0]
    return metrics, best


def train_models(
    X,
    y,
    X_test,
    X_external,
    y_external,
    categorical_cols,
    splits,
    seed,
    profile_name,
    external_weight,
    log_period,
):
    oof_prediction = np.zeros(len(X), dtype=float)
    test_prediction = (
        np.zeros(len(X_test), dtype=float)
        if X_test is not None
        else None
    )
    fold_records = []
    importance_frames = []

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        X_train_comp = X.iloc[train_idx]
        y_train_comp = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        X_train = pd.concat([X_train_comp, X_external], ignore_index=True)
        y_train = pd.concat([y_train_comp, y_external], ignore_index=True)
        sample_weight = np.concatenate(
            [
                np.ones(len(X_train_comp), dtype=float),
                np.full(len(X_external), external_weight, dtype=float),
            ]
        )

        model = lgb.LGBMClassifier(**base_params(seed + fold, profile_name))
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=300, verbose=False),
                lgb.log_evaluation(period=log_period),
            ],
        )
        valid_prediction = model.predict_proba(
            X_valid,
            num_iteration=model.best_iteration_,
        )[:, 1]
        oof_prediction[valid_idx] = valid_prediction
        if X_test is not None:
            test_prediction += (
                model.predict_proba(
                    X_test,
                    num_iteration=model.best_iteration_,
                )[:, 1]
                / len(splits)
            )

        gain = model.booster_.feature_importance(importance_type="gain")
        gain_total = gain.sum()
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "gain_percent": gain / gain_total * 100 if gain_total else 0.0,
                    "seed": seed,
                    "fold": fold,
                }
            )
        )
        fold_auc = roc_auc_score(y_valid, valid_prediction)
        fold_records.append(
            {
                "seed": seed,
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": model.best_iteration_,
                "profile": profile_name,
                "external_weight": external_weight,
            }
        )
        print(
            f"  seed={seed} fold={fold}/{len(splits)} "
            f"AUC={fold_auc:.6f}, best_iteration={model.best_iteration_}"
        )
        del model, X_train, y_train, X_valid, y_valid
        gc.collect()

    return {
        "oof_prediction": np.clip(oof_prediction, 0.0, 1.0),
        "test_prediction": (
            np.clip(test_prediction, 0.0, 1.0)
            if test_prediction is not None
            else None
        ),
        "fold_records": fold_records,
        "importance": pd.concat(importance_frames, ignore_index=True),
    }


def get_feature_cache(train, test, external, cache, feature_set_name):
    if feature_set_name not in cache:
        cache[feature_set_name] = build_features(
            train,
            test,
            external,
            feature_set_name,
        )
    return cache[feature_set_name]


def run_search(train, test, external, y, y_external, anchor):
    feature_cache = {}
    search_records = []
    search_fold_records = []
    splits = list(
        StratifiedKFold(
            n_splits=SEARCH_SPLITS,
            shuffle=True,
            random_state=SEARCH_SEED,
        ).split(train, y)
    )

    print("\n" + "=" * 70)
    print("20th blend-oriented LightGBM search")
    print("=" * 70)
    print(f"Candidates: {len(SEARCH_VARIANTS)}")
    print(f"Search CV: {SEARCH_SPLITS}-fold, seed={SEARCH_SEED}")

    for index, variant in enumerate(SEARCH_VARIANTS, start=1):
        print("\n" + "-" * 70)
        print(f"Candidate {index}/{len(SEARCH_VARIANTS)}: {variant['name']}")
        print("-" * 70)
        X, _, X_external, categorical_cols = get_feature_cache(
            train,
            test,
            external,
            feature_cache,
            variant["feature_set"],
        )
        result = train_models(
            X,
            y,
            None,
            X_external,
            y_external,
            categorical_cols,
            splits,
            SEARCH_SEED,
            variant["profile"],
            variant["external_weight"],
            log_period=0,
        )
        lightgbm_auc = roc_auc_score(y, result["oof_prediction"])
        blend_metrics, best_blend = evaluate_blend(
            y,
            anchor["oof_prediction"],
            result["oof_prediction"],
        )
        correlation = np.corrcoef(
            anchor["oof_prediction"],
            result["oof_prediction"],
        )[0, 1]
        search_records.append(
            {
                **variant,
                "lightgbm_oof_auc": lightgbm_auc,
                "realmlp_correlation": correlation,
                "best_blend_auc": best_blend["blend_auc"],
                "best_lightgbm_weight": best_blend["lightgbm_weight"],
                "best_realmlp_weight": best_blend["realmlp_weight"],
            }
        )
        for row in result["fold_records"]:
            search_fold_records.append({"variant": variant["name"], **row})
        print(
            f"LightGBM OOF={lightgbm_auc:.6f}, "
            f"corr={correlation:.6f}, "
            f"best blend={best_blend['blend_auc']:.6f} "
            f"at LGBM weight={best_blend['lightgbm_weight']:.2f}"
        )

        pd.DataFrame(search_records).sort_values(
            "best_blend_auc",
            ascending=False,
        ).to_csv(DIAGNOSTIC_DIR / "search_results_partial.csv", index=False)
        blend_metrics.to_csv(
            DIAGNOSTIC_DIR / f"search_blend_{variant['name']}.csv",
            index=False,
        )

    search_results = pd.DataFrame(search_records).sort_values(
        ["best_blend_auc", "lightgbm_oof_auc"],
        ascending=False,
    ).reset_index(drop=True)
    search_fold_metrics = pd.DataFrame(search_fold_records)
    print("\n" + "=" * 70)
    print("Search ranking")
    print("=" * 70)
    print(search_results.to_string(index=False))
    return search_results, search_fold_metrics


def train_final(
    train,
    test,
    external,
    y,
    y_external,
    anchor,
    best_variant,
):
    X, X_test, X_external, categorical_cols = build_features(
        train,
        test,
        external,
        best_variant["feature_set"],
    )
    all_oof = []
    all_test = []
    fold_records = []
    importance_frames = []
    seed_records = []

    print("\n" + "=" * 70)
    print(f"Final training: {best_variant['name']}")
    print("=" * 70)

    for seed in FINAL_SEEDS:
        print(f"\nFinal seed: {seed}")
        splits = list(
            StratifiedKFold(
                n_splits=FINAL_SPLITS,
                shuffle=True,
                random_state=seed,
            ).split(X, y)
        )
        result = train_models(
            X,
            y,
            X_test,
            X_external,
            y_external,
            categorical_cols,
            splits,
            seed,
            best_variant["profile"],
            float(best_variant["external_weight"]),
            log_period=500,
        )
        seed_auc = roc_auc_score(y, result["oof_prediction"])
        seed_records.append({"seed": seed, "oof_auc": seed_auc})
        all_oof.append(result["oof_prediction"])
        all_test.append(result["test_prediction"])
        fold_records.extend(result["fold_records"])
        importance_frames.append(result["importance"])
        print(f"Final seed {seed}: OOF AUC={seed_auc:.6f}")

    oof_prediction = np.mean(all_oof, axis=0)
    test_prediction = np.mean(all_test, axis=0)
    lightgbm_auc = roc_auc_score(y, oof_prediction)
    blend_metrics, best_blend = evaluate_blend(
        y,
        anchor["oof_prediction"],
        oof_prediction,
    )
    blended_test_prediction = (
        best_blend["realmlp_weight"] * anchor["test_prediction"]
        + best_blend["lightgbm_weight"] * test_prediction
    )
    importance_summary = (
        pd.concat(importance_frames, ignore_index=True)
        .groupby("feature", as_index=False)
        .agg(
            gain_mean=("gain_percent", "mean"),
            gain_std=("gain_percent", "std"),
        )
        .sort_values("gain_mean", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "oof_prediction": np.clip(oof_prediction, 0.0, 1.0),
        "test_prediction": np.clip(test_prediction, 0.0, 1.0),
        "seed_oof_predictions": {
            f"seed_{seed}": prediction
            for seed, prediction in zip(FINAL_SEEDS, all_oof)
        },
        "fold_metrics": pd.DataFrame(fold_records),
        "seed_metrics": pd.DataFrame(seed_records),
        "importance_summary": importance_summary,
        "lightgbm_auc": lightgbm_auc,
        "blend_metrics": blend_metrics,
        "best_blend": best_blend,
        "blended_test_prediction": np.clip(blended_test_prediction, 0.0, 1.0),
    }


def save_results(
    train,
    test,
    sample,
    anchor,
    search_results,
    search_fold_metrics,
    best_variant,
    result,
):
    search_results.to_csv(DIAGNOSTIC_DIR / "search_results.csv", index=False)
    search_fold_metrics.to_csv(
        DIAGNOSTIC_DIR / "search_fold_metrics.csv",
        index=False,
    )
    result["fold_metrics"].to_csv(
        DIAGNOSTIC_DIR / "fold_metrics.csv",
        index=False,
    )
    result["seed_metrics"].to_csv(
        DIAGNOSTIC_DIR / "seed_metrics.csv",
        index=False,
    )
    result["importance_summary"].to_csv(
        DIAGNOSTIC_DIR / "feature_importance.csv",
        index=False,
    )
    result["blend_metrics"].to_csv(
        DIAGNOSTIC_DIR / "final_blend_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET_COL: train[TARGET_COL],
            OOF_PRED_COL: result["oof_prediction"],
        }
    ).to_csv(DIAGNOSTIC_DIR / "oof_predictions.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET_COL: train[TARGET_COL],
            **result["seed_oof_predictions"],
        }
    ).to_csv(DIAGNOSTIC_DIR / "seed_oof_predictions.csv", index=False)

    best_blend = result["best_blend"]
    run_summary = {
        "anchor_model": anchor["name"],
        "selected_variant": best_variant["name"],
        "selected_feature_set": best_variant["feature_set"],
        "selected_profile": best_variant["profile"],
        "external_weight": float(best_variant["external_weight"]),
        "lightgbm_oof_auc": float(result["lightgbm_auc"]),
        "blend_oof_auc": float(best_blend["blend_auc"]),
        "realmlp_weight": float(best_blend["realmlp_weight"]),
        "lightgbm_weight": float(best_blend["lightgbm_weight"]),
    }
    pd.DataFrame(
        {
            "metric": list(run_summary.keys()),
            "value": list(run_summary.values()),
        }
    ).to_csv(DIAGNOSTIC_DIR / "run_summary.csv", index=False)
    with (DIAGNOSTIC_DIR / "best_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_summary, file, indent=2)

    standalone = sample.copy()
    standalone[ID_COL] = test[ID_COL].values
    standalone[TARGET_COL] = result["test_prediction"]
    standalone.to_csv(STANDALONE_OUTPUT_PATH, index=False)

    blended = sample.copy()
    blended[ID_COL] = test[ID_COL].values
    blended[TARGET_COL] = result["blended_test_prediction"]
    blended.to_csv(BLEND_OUTPUT_PATH, index=False)

    print("\n" + "=" * 70)
    print("20th pre-submission diagnostics")
    print("=" * 70)
    print(f"Anchor: {anchor['name']}")
    print(f"Selected variant: {best_variant['name']}")
    print(f"Standalone LightGBM OOF AUC: {result['lightgbm_auc']:.6f}")
    print(
        "Best probability blend OOF AUC: "
        f"{best_blend['blend_auc']:.6f} "
        f"({best_blend['realmlp_weight']:.2f}*{anchor['name']} + "
        f"{best_blend['lightgbm_weight']:.2f}*lightgbm_20)"
    )
    print(f"\nSaved standalone submission: {STANDALONE_OUTPUT_PATH}")
    print(f"Saved blend submission: {BLEND_OUTPUT_PATH}")
    print(f"Saved diagnostics: {DIAGNOSTIC_DIR}")


def main():
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    train, test, sample, external = load_data()
    y = train[TARGET_COL].astype(int).reset_index(drop=True)
    y_external = external[TARGET_COL].astype(int).reset_index(drop=True)
    anchor = load_anchor(train, test)

    print(f"Competition train shape: {train.shape}")
    print(f"External train shape: {external.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Anchor OOF AUC: {roc_auc_score(y, anchor['oof_prediction']):.6f}")

    search_results, search_fold_metrics = run_search(
        train,
        test,
        external,
        y,
        y_external,
        anchor,
    )
    best_variant = search_results.iloc[0].to_dict()
    result = train_final(
        train,
        test,
        external,
        y,
        y_external,
        anchor,
        best_variant,
    )
    save_results(
        train,
        test,
        sample,
        anchor,
        search_results,
        search_fold_metrics,
        best_variant,
        result,
    )


if __name__ == "__main__":
    main()
