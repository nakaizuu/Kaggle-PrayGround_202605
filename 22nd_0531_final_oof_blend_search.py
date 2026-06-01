from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Required package is missing. Install dependencies with:\n"
        "pip install scikit-learn pandas numpy"
    ) from exc


DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
DIAGNOSTIC_DIR = DATA_DIR / "outputs" / "22nd_final_oof_blend_search"

ID_COL = "id"
TARGET_COL = "PitNextLap"
OOF_PRED_COL = "oof_prediction"
EPSILON = 1e-7
PAIR_PROBABILITY_MAX_WEIGHT = 0.40
PAIR_RANK_MAX_WEIGHT = 0.20
MAX_TOTAL_SUPPORT_WEIGHT = 0.45
PAIR_STEP = 0.01
COARSE_STEP = 0.05
FINE_STEP = 0.01
TOP_TRIPLE_SUPPORT_MODELS = 4
TRIPLE_FINE_RADIUS = 0.03
STABILITY_SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]
NEAR_BEST_AUC_DELTA = 0.00005

MODEL_SPECS = {
    "realmlp_19": {
        "label": "19th GPU RealMLP 6-epoch seed ensemble",
        "family": "realmlp",
        "submission": DATA_DIR
        / "submission_19th_gpu_realmlp_6epoch_seed_ensemble.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "19th_gpu_realmlp_6epoch_seed_ensemble"
            / "oof_predictions.csv",
            DATA_DIR / "outputs" / "19th_pytorch_improve" / "oof_predictions.csv",
        ],
    },
    "realmlp_18": {
        "label": "18th GPU RealMLP seed ensemble",
        "family": "realmlp",
        "submission": DATA_DIR / "submission_18th_gpu_realmlp_seed_ensemble.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "18th_gpu_realmlp_seed_ensemble"
            / "oof_predictions.csv",
        ],
    },
    "realmlp_17": {
        "label": "17th GPU RealMLP reference",
        "family": "realmlp",
        "submission": DATA_DIR / "submission_17th_gpu_realmlp_reference.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "17th_gpu_realmlp_reference"
            / "oof_predictions.csv",
            DATA_DIR / "outputs" / "17th_super_pytorch" / "oof_predictions.csv",
        ],
    },
    "lightgbm_20": {
        "label": "20th blend-oriented LightGBM",
        "family": "lightgbm",
        "submission": DATA_DIR / "submission_20th_blend_optimized_lightgbm.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "20th_blend_optimized_lightgbm"
            / "oof_predictions.csv",
        ],
    },
    "lightgbm_7": {
        "label": "7th external-data LightGBM ensemble",
        "family": "lightgbm",
        "submission": DATA_DIR / "submission_lightgbm_external_seed_ensemble.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "7th_external_data_seed_ensemble"
            / "oof_predictions.csv",
            DATA_DIR
            / "outputs"
            / "7th_external_data_seed_ensemble"
            / "7th_lightgbm_oof_predictions.csv",
        ],
    },
    "pytorch_21": {
        "label": "21st GPU PyTorch residual complement",
        "family": "pytorch",
        "submission": DATA_DIR
        / "submission_21st_gpu_pytorch_residual_complement.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "21st_gpu_pytorch_residual_complement"
            / "oof_predictions.csv",
        ],
    },
    "pytorch_15": {
        "label": "15th GPU PyTorch seed ensemble",
        "family": "pytorch",
        "submission": DATA_DIR / "submission_15th_gpu_pytorch_seed_ensemble.csv",
        "oof_candidates": [
            DATA_DIR
            / "outputs"
            / "15th_gpu_pytorch_seed_ensemble"
            / "oof_predictions.csv",
            DATA_DIR / "outputs" / "15th_advanced_pytorch" / "oof_predictions.csv",
        ],
    },
}

OUTPUT_PATHS = {
    "best_probability": DATA_DIR / "submission_22nd_best_probability_blend.csv",
    "conservative_probability": DATA_DIR
    / "submission_22nd_conservative_probability_blend.csv",
    "best_rank_remap": DATA_DIR / "submission_22nd_best_rank_remap_blend.csv",
    "best_logit_probability": DATA_DIR
    / "submission_22nd_best_logit_probability_blend.csv",
    "best_logit_rank_remap": DATA_DIR
    / "submission_22nd_best_logit_rank_remap_blend.csv",
    "best_overall": DATA_DIR / "submission_22nd_best_overall_blend.csv",
}


def find_existing_path(candidates):
    return next((path for path in candidates if path.exists()), None)


def read_oof(path, model_name):
    frame = pd.read_csv(path)
    required_cols = {ID_COL, TARGET_COL, OOF_PRED_COL}
    missing_cols = sorted(required_cols - set(frame.columns))
    if missing_cols:
        raise ValueError(f"{model_name} OOF is missing columns: {missing_cols}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{model_name} OOF contains duplicated IDs.")
    return frame[[ID_COL, TARGET_COL, OOF_PRED_COL]].copy()


def read_submission(path, model_name):
    frame = pd.read_csv(path)
    required_cols = {ID_COL, TARGET_COL}
    missing_cols = sorted(required_cols - set(frame.columns))
    if missing_cols:
        raise ValueError(f"{model_name} submission is missing columns: {missing_cols}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{model_name} submission contains duplicated IDs.")
    return frame[[ID_COL, TARGET_COL]].copy()


def load_prediction_matrices():
    train = pd.read_csv(TRAIN_PATH, usecols=[ID_COL, TARGET_COL])
    sample = pd.read_csv(SAMPLE_PATH, usecols=[ID_COL])
    loaded_models = []
    skipped_models = []
    oof_matrix = train.copy()
    test_matrix = sample.copy()

    for model_name, spec in MODEL_SPECS.items():
        oof_path = find_existing_path(spec["oof_candidates"])
        submission_path = spec["submission"]
        if oof_path is None or not submission_path.exists():
            missing = []
            if oof_path is None:
                missing.append("OOF")
            if not submission_path.exists():
                missing.append("submission")
            skipped_models.append(f"{model_name}: missing {', '.join(missing)}")
            continue

        model_oof = read_oof(oof_path, model_name).rename(
            columns={OOF_PRED_COL: model_name}
        )
        model_test = read_submission(submission_path, model_name).rename(
            columns={TARGET_COL: model_name}
        )
        oof_matrix = oof_matrix.merge(
            model_oof,
            on=[ID_COL, TARGET_COL],
            how="left",
            validate="one_to_one",
        )
        test_matrix = test_matrix.merge(
            model_test,
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
        if oof_matrix[model_name].isna().any():
            raise ValueError(f"{model_name} OOF could not be aligned by ID.")
        if test_matrix[model_name].isna().any():
            raise ValueError(f"{model_name} submission could not be aligned by ID.")

        loaded_models.append(model_name)
        print(
            f"Loaded {model_name}: "
            f"OOF={oof_path.relative_to(DATA_DIR)}, "
            f"submission={submission_path.name}"
        )

    return oof_matrix, test_matrix, loaded_models, skipped_models


def clip_probability(values):
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1.0 - EPSILON)


def normalized_rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def logit(values):
    values = clip_probability(values)
    return np.log(values / (1.0 - values))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def rank_remap(anchor_values, ordering_signal):
    order = np.argsort(ordering_signal, kind="mergesort")
    remapped = np.empty_like(anchor_values, dtype=float)
    remapped[order] = np.sort(anchor_values)
    return clip_probability(remapped)


def build_context(frame, model_names):
    probability = {
        model_name: clip_probability(frame[model_name].to_numpy())
        for model_name in model_names
    }
    ranks = {
        model_name: normalized_rank(probability[model_name])
        for model_name in model_names
    }
    return {
        "probability": probability,
        "logit_probability": {
            model_name: logit(probability[model_name])
            for model_name in model_names
        },
        "rank": ranks,
        "logit_rank": {
            model_name: logit(ranks[model_name])
            for model_name in model_names
        },
    }


def normalized_weights(weights):
    positive_weights = {
        model_name: float(weight)
        for model_name, weight in weights.items()
        if weight > 1e-12
    }
    total = sum(positive_weights.values())
    return {
        model_name: weight / total
        for model_name, weight in positive_weights.items()
    }


def select_anchor(model_names, oof_matrix, target):
    candidates = [
        model_name
        for model_name in model_names
        if MODEL_SPECS[model_name]["family"] == "realmlp"
    ]
    if not candidates:
        raise ValueError("No RealMLP anchor was loaded.")
    aucs = {
        model_name: roc_auc_score(target, oof_matrix[model_name])
        for model_name in candidates
    }
    anchor_model = max(candidates, key=lambda model_name: aucs[model_name])
    return anchor_model, aucs


def scaled_support_weights(weights, anchor_model, scale):
    support_weights = {
        model_name: weight * scale
        for model_name, weight in weights.items()
        if model_name != anchor_model
    }
    support_total = sum(support_weights.values())
    if support_total >= 0.95:
        return None
    return normalized_weights(
        {
            anchor_model: 1.0 - support_total,
            **support_weights,
        }
    )


def predict_with_method(context, method, weights, anchor_model):
    weights = normalized_weights(weights)
    if method == "probability":
        return clip_probability(
            sum(
                weight * context["probability"][model_name]
                for model_name, weight in weights.items()
            )
        )
    if method == "logit_probability":
        return clip_probability(
            sigmoid(
                sum(
                    weight * context["logit_probability"][model_name]
                    for model_name, weight in weights.items()
                )
            )
        )
    if method == "rank_remap":
        ordering_signal = sum(
            weight * context["rank"][model_name]
            for model_name, weight in weights.items()
        )
        return rank_remap(
            context["probability"][anchor_model],
            ordering_signal,
        )
    if method == "logit_rank_remap":
        ordering_signal = sigmoid(
            sum(
                weight * context["logit_rank"][model_name]
                for model_name, weight in weights.items()
            )
        )
        return rank_remap(
            context["probability"][anchor_model],
            ordering_signal,
        )
    raise ValueError(f"Unknown blend method: {method}")


def weights_text(weights):
    weights = normalized_weights(weights)
    return " + ".join(
        f"{weight:.2f}*{model_name}"
        for model_name, weight in sorted(
            weights.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


def evaluate_candidate(oof_context, target, anchor_model, method, weights, stage):
    weights = normalized_weights(weights)
    prediction = predict_with_method(oof_context, method, weights, anchor_model)
    support_total = sum(
        weight
        for model_name, weight in weights.items()
        if model_name != anchor_model
    )
    return {
        "method": method,
        "stage": stage,
        "formula": weights_text(weights),
        "weights_json": json.dumps(weights, sort_keys=True),
        "model_count": len(weights),
        "support_weight": support_total,
        "auc": roc_auc_score(target, prediction),
    }


def add_stability_metrics(row, oof_context, target, anchor_model):
    record = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    weights = json.loads(record["weights_json"])
    neighborhood_aucs = []
    for scale in STABILITY_SCALES:
        scaled_weights = scaled_support_weights(weights, anchor_model, scale)
        if scaled_weights is None:
            continue
        prediction = predict_with_method(
            oof_context,
            record["method"],
            scaled_weights,
            anchor_model,
        )
        neighborhood_aucs.append(roc_auc_score(target, prediction))
    record["neighborhood_mean_auc"] = float(np.mean(neighborhood_aucs))
    record["neighborhood_min_auc"] = float(np.min(neighborhood_aucs))
    return pd.Series(record)


def scan_pair_blends(oof_context, target, anchor_model, support_models):
    records = []
    for support_model in support_models:
        for support_weight in np.arange(
            0.0,
            PAIR_PROBABILITY_MAX_WEIGHT + 0.001,
            PAIR_STEP,
        ):
            weights = {
                anchor_model: 1.0 - support_weight,
                support_model: support_weight,
            }
            for method in ["probability", "logit_probability"]:
                records.append(
                    evaluate_candidate(
                        oof_context,
                        target,
                        anchor_model,
                        method,
                        weights,
                        stage=f"pair_{support_model}",
                    )
                )

        for support_weight in np.arange(
            0.0,
            PAIR_RANK_MAX_WEIGHT + 0.001,
            PAIR_STEP,
        ):
            weights = {
                anchor_model: 1.0 - support_weight,
                support_model: support_weight,
            }
            for method in ["rank_remap", "logit_rank_remap"]:
                records.append(
                    evaluate_candidate(
                        oof_context,
                        target,
                        anchor_model,
                        method,
                        weights,
                        stage=f"pair_{support_model}",
                    )
                )

    return pd.DataFrame(records)


def scan_coarse_probability_blends(
    oof_context,
    target,
    anchor_model,
    support_models,
):
    records = []
    for first_support, second_support in combinations(support_models, 2):
        for first_weight in np.arange(
            COARSE_STEP,
            MAX_TOTAL_SUPPORT_WEIGHT + 0.001,
            COARSE_STEP,
        ):
            for second_weight in np.arange(
                COARSE_STEP,
                MAX_TOTAL_SUPPORT_WEIGHT - first_weight + 0.001,
                COARSE_STEP,
            ):
                support_total = first_weight + second_weight
                weights = {
                    anchor_model: 1.0 - support_total,
                    first_support: first_weight,
                    second_support: second_weight,
                }
                records.append(
                    evaluate_candidate(
                        oof_context,
                        target,
                        anchor_model,
                        "probability",
                        weights,
                        stage="coarse_multi_probability",
                    )
                )
    return pd.DataFrame(records)


def scan_fine_probability_blends(
    oof_context,
    target,
    anchor_model,
    coarse_metrics,
):
    if coarse_metrics.empty:
        return coarse_metrics

    best_weights = json.loads(
        coarse_metrics.sort_values("auc", ascending=False).iloc[0]["weights_json"]
    )
    support_models = [
        model_name
        for model_name in best_weights
        if model_name != anchor_model
    ]
    if len(support_models) != 2:
        return pd.DataFrame()

    first_support, second_support = support_models
    first_center = best_weights[first_support]
    second_center = best_weights[second_support]
    first_values = np.arange(
        max(FINE_STEP, first_center - 0.05),
        min(MAX_TOTAL_SUPPORT_WEIGHT, first_center + 0.05) + 0.001,
        FINE_STEP,
    )
    second_values = np.arange(
        max(FINE_STEP, second_center - 0.05),
        min(MAX_TOTAL_SUPPORT_WEIGHT, second_center + 0.05) + 0.001,
        FINE_STEP,
    )
    records = []
    for first_weight in first_values:
        for second_weight in second_values:
            support_total = first_weight + second_weight
            if support_total > MAX_TOTAL_SUPPORT_WEIGHT + 1e-9:
                continue
            weights = {
                anchor_model: 1.0 - support_total,
                first_support: float(first_weight),
                second_support: float(second_weight),
            }
            records.append(
                evaluate_candidate(
                    oof_context,
                    target,
                    anchor_model,
                    "probability",
                    weights,
                    stage="fine_multi_probability",
                )
            )
    return pd.DataFrame(records)


def select_top_support_models(pair_metrics):
    probability_pairs = pair_metrics[
        pair_metrics["method"].eq("probability")
    ].copy()
    probability_pairs["support_model"] = probability_pairs["stage"].str.replace(
        "pair_",
        "",
        regex=False,
    )
    return (
        probability_pairs.sort_values("auc", ascending=False)
        .drop_duplicates("support_model")
        .head(TOP_TRIPLE_SUPPORT_MODELS)["support_model"]
        .tolist()
    )


def scan_coarse_three_support_probability_blends(
    oof_context,
    target,
    anchor_model,
    pair_metrics,
):
    records = []
    top_support_models = select_top_support_models(pair_metrics)
    for first_support, second_support, third_support in combinations(
        top_support_models,
        3,
    ):
        for first_weight in np.arange(
            COARSE_STEP,
            MAX_TOTAL_SUPPORT_WEIGHT + 0.001,
            COARSE_STEP,
        ):
            for second_weight in np.arange(
                COARSE_STEP,
                MAX_TOTAL_SUPPORT_WEIGHT - first_weight + 0.001,
                COARSE_STEP,
            ):
                for third_weight in np.arange(
                    COARSE_STEP,
                    MAX_TOTAL_SUPPORT_WEIGHT
                    - first_weight
                    - second_weight
                    + 0.001,
                    COARSE_STEP,
                ):
                    support_total = (
                        first_weight + second_weight + third_weight
                    )
                    weights = {
                        anchor_model: 1.0 - support_total,
                        first_support: float(first_weight),
                        second_support: float(second_weight),
                        third_support: float(third_weight),
                    }
                    records.append(
                        evaluate_candidate(
                            oof_context,
                            target,
                            anchor_model,
                            "probability",
                            weights,
                            stage="coarse_three_support_probability",
                        )
                    )
    return pd.DataFrame(records)


def scan_fine_three_support_probability_blends(
    oof_context,
    target,
    anchor_model,
    coarse_metrics,
):
    if coarse_metrics.empty:
        return coarse_metrics

    best_weights = json.loads(
        coarse_metrics.sort_values("auc", ascending=False).iloc[0]["weights_json"]
    )
    support_models = [
        model_name
        for model_name in best_weights
        if model_name != anchor_model
    ]
    if len(support_models) != 3:
        return pd.DataFrame()

    support_values = []
    for support_model in support_models:
        center = best_weights[support_model]
        support_values.append(
            np.arange(
                max(FINE_STEP, center - TRIPLE_FINE_RADIUS),
                min(
                    MAX_TOTAL_SUPPORT_WEIGHT,
                    center + TRIPLE_FINE_RADIUS,
                )
                + 0.001,
                FINE_STEP,
            )
        )

    records = []
    for first_weight in support_values[0]:
        for second_weight in support_values[1]:
            for third_weight in support_values[2]:
                support_total = first_weight + second_weight + third_weight
                if support_total > MAX_TOTAL_SUPPORT_WEIGHT + 1e-9:
                    continue
                weights = {
                    anchor_model: 1.0 - support_total,
                    support_models[0]: float(first_weight),
                    support_models[1]: float(second_weight),
                    support_models[2]: float(third_weight),
                }
                records.append(
                    evaluate_candidate(
                        oof_context,
                        target,
                        anchor_model,
                        "probability",
                        weights,
                        stage="fine_three_support_probability",
                    )
                )
    return pd.DataFrame(records)


def select_best(metrics, method, oof_context, target, anchor_model):
    method_metrics = metrics[metrics["method"].eq(method)]
    if method_metrics.empty:
        return None
    best = method_metrics.sort_values(
        ["auc", "model_count", "support_weight"],
        ascending=[False, True, True],
    ).iloc[0]
    return add_stability_metrics(best, oof_context, target, anchor_model)


def select_conservative_probability(
    probability_metrics,
    oof_context,
    target,
    anchor_model,
):
    best_auc = probability_metrics["auc"].max()
    near_best = probability_metrics[
        probability_metrics["auc"] >= best_auc - NEAR_BEST_AUC_DELTA
    ]
    stable_rows = pd.DataFrame(
        [
            add_stability_metrics(row, oof_context, target, anchor_model)
            for _, row in near_best.iterrows()
        ]
    )
    return stable_rows.sort_values(
        [
            "neighborhood_mean_auc",
            "neighborhood_min_auc",
            "auc",
            "model_count",
            "support_weight",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def save_submission(
    test_matrix,
    test_context,
    anchor_model,
    finalist_name,
    finalist_row,
):
    weights = json.loads(finalist_row["weights_json"])
    prediction = predict_with_method(
        test_context,
        finalist_row["method"],
        weights,
        anchor_model,
    )
    submission = test_matrix[[ID_COL]].copy()
    submission[TARGET_COL] = clip_probability(prediction)
    output_path = OUTPUT_PATHS[finalist_name]
    submission.to_csv(output_path, index=False)
    print(f"\nSaved {finalist_name}: {output_path.name}")
    print(
        f"OOF AUC={finalist_row['auc']:.6f}, "
        f"method={finalist_row['method']}, "
        f"formula={finalist_row['formula']}"
    )


def main():
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    oof_matrix, test_matrix, model_names, skipped_models = (
        load_prediction_matrices()
    )
    target = oof_matrix[TARGET_COL].to_numpy()
    anchor_model, anchor_aucs = select_anchor(model_names, oof_matrix, target)
    support_models = [
        model_name for model_name in model_names if model_name != anchor_model
    ]
    if not support_models:
        raise ValueError("No support models were loaded.")

    oof_context = build_context(oof_matrix, model_names)
    test_context = build_context(test_matrix, model_names)
    model_auc = pd.DataFrame(
        [
            {
                "model": model_name,
                "label": MODEL_SPECS[model_name]["label"],
                "family": MODEL_SPECS[model_name]["family"],
                "auc": roc_auc_score(target, oof_matrix[model_name]),
                "prediction_mean": oof_matrix[model_name].mean(),
                "prediction_std": oof_matrix[model_name].std(),
            }
            for model_name in model_names
        ]
    ).sort_values("auc", ascending=False)
    pearson_correlations = oof_matrix[model_names].corr(method="pearson")
    spearman_correlations = oof_matrix[model_names].corr(method="spearman")

    print("\n" + "=" * 70)
    print("22nd final OOF blend search")
    print("=" * 70)
    print(f"Loaded models: {model_names}")
    if skipped_models:
        print(f"Skipped models: {skipped_models}")
    print(f"Selected anchor: {anchor_model}")
    print(f"RealMLP anchor candidates: {anchor_aucs}")
    print("\nBase model OOF AUC:")
    print(model_auc.to_string(index=False))
    print("\nPearson OOF prediction correlations:")
    print(pearson_correlations.to_string())

    pair_metrics = scan_pair_blends(
        oof_context,
        target,
        anchor_model,
        support_models,
    )
    coarse_probability_metrics = scan_coarse_probability_blends(
        oof_context,
        target,
        anchor_model,
        support_models,
    )
    fine_probability_metrics = scan_fine_probability_blends(
        oof_context,
        target,
        anchor_model,
        coarse_probability_metrics,
    )
    coarse_three_support_metrics = scan_coarse_three_support_probability_blends(
        oof_context,
        target,
        anchor_model,
        pair_metrics,
    )
    fine_three_support_metrics = scan_fine_three_support_probability_blends(
        oof_context,
        target,
        anchor_model,
        coarse_three_support_metrics,
    )
    probability_metrics = pd.concat(
        [
            pair_metrics[pair_metrics["method"].eq("probability")],
            coarse_probability_metrics,
            fine_probability_metrics,
            coarse_three_support_metrics,
            fine_three_support_metrics,
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["method", "weights_json"])
    all_metrics = pd.concat(
        [
            probability_metrics,
            pair_metrics[~pair_metrics["method"].eq("probability")],
        ],
        ignore_index=True,
    )

    finalists = {
        "best_probability": select_best(
            probability_metrics,
            "probability",
            oof_context,
            target,
            anchor_model,
        ),
        "conservative_probability": select_conservative_probability(
            probability_metrics,
            oof_context,
            target,
            anchor_model,
        ),
        "best_rank_remap": select_best(
            all_metrics,
            "rank_remap",
            oof_context,
            target,
            anchor_model,
        ),
        "best_logit_probability": select_best(
            all_metrics,
            "logit_probability",
            oof_context,
            target,
            anchor_model,
        ),
        "best_logit_rank_remap": select_best(
            all_metrics,
            "logit_rank_remap",
            oof_context,
            target,
            anchor_model,
        ),
    }
    finalist_frame = pd.DataFrame(
        [
            {"candidate": candidate_name, **finalist.to_dict()}
            for candidate_name, finalist in finalists.items()
            if finalist is not None
        ]
    )
    best_overall = finalist_frame.sort_values(
        [
            "auc",
            "neighborhood_mean_auc",
            "neighborhood_min_auc",
            "model_count",
            "support_weight",
        ],
        ascending=[False, False, False, True, True],
    ).iloc[0]
    finalists["best_overall"] = best_overall

    model_auc.to_csv(DIAGNOSTIC_DIR / "model_auc.csv", index=False)
    pearson_correlations.to_csv(DIAGNOSTIC_DIR / "pearson_correlations.csv")
    spearman_correlations.to_csv(DIAGNOSTIC_DIR / "spearman_correlations.csv")
    pair_metrics.sort_values("auc", ascending=False).to_csv(
        DIAGNOSTIC_DIR / "pair_blend_metrics.csv",
        index=False,
    )
    probability_metrics.sort_values("auc", ascending=False).to_csv(
        DIAGNOSTIC_DIR / "probability_blend_metrics.csv",
        index=False,
    )
    finalist_frame.sort_values("auc", ascending=False).to_csv(
        DIAGNOSTIC_DIR / "finalists.csv",
        index=False,
    )
    with (DIAGNOSTIC_DIR / "loaded_models.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "loaded_models": model_names,
                "skipped_models": skipped_models,
                "selected_anchor": anchor_model,
                "realmlp_anchor_auc": anchor_aucs,
            },
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("Top probability blends")
    print("=" * 70)
    print(
        probability_metrics.sort_values("auc", ascending=False)
        .head(15)
        .to_string(index=False)
    )
    print("\n" + "=" * 70)
    print("Final submission candidates")
    print("=" * 70)
    print(
        finalist_frame.sort_values("auc", ascending=False).to_string(index=False)
    )

    for finalist_name, finalist in finalists.items():
        save_submission(
            test_matrix,
            test_context,
            anchor_model,
            finalist_name,
            finalist,
        )

    print(f"\nSaved diagnostics: {DIAGNOSTIC_DIR}")
    print(
        "Review finalists.csv before submitting. "
        "Prefer the conservative probability candidate when tiny OOF gains "
        "require noticeably sharper weights."
    )


if __name__ == "__main__":
    main()
