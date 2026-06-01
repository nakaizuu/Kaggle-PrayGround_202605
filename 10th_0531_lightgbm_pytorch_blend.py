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
LIGHTGBM_SUBMISSION_PATH = DATA_DIR / "submission_lightgbm_external_seed_ensemble.csv"
PYTORCH_SUBMISSION_PATH = DATA_DIR / "submission_pytorch_tabular_mlp.csv"
LIGHTGBM_OOF_CANDIDATES = [
    DATA_DIR
    / "outputs"
    / "7th_external_data_seed_ensemble"
    / "oof_predictions.csv",
    DATA_DIR
    / "outputs"
    / "7th_external_data_seed_ensemble"
    / "7th_lightgbm_oof_predictions.csv",
]
PYTORCH_OOF_PATH = DATA_DIR / "outputs" / "9th_easy_pytorch" / "oof_predictions.csv"
OUTPUT_PROBABILITY_PATH = (
    DATA_DIR / "submission_10th_lgbm70_pytorch30_probability_blend.csv"
)
OUTPUT_RANK_PATH = DATA_DIR / "submission_10th_lgbm70_pytorch30_rank_blend.csv"
DIAGNOSTIC_DIR = DATA_DIR / "outputs" / "10th_lgbm_pytorch_blend"

ID_COL = "id"
TARGET_COL = "PitNextLap"
OOF_PRED_COL = "oof_prediction"

# The 0.70 / 0.30 ratio is deliberately rounded. The coarse OOF scan from the
# 9th experiment peaked here, and a simple ratio is less likely to overfit OOF.
LIGHTGBM_WEIGHT = 0.70
PYTORCH_WEIGHT = 1.0 - LIGHTGBM_WEIGHT


def read_prediction(path, prediction_col, model_name):
    if not path.exists():
        raise FileNotFoundError(f"{model_name} file was not found: {path}")

    frame = pd.read_csv(path)
    required_cols = {ID_COL, prediction_col}
    missing_cols = sorted(required_cols - set(frame.columns))
    if missing_cols:
        raise ValueError(f"{model_name} file is missing columns: {missing_cols}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{model_name} file contains duplicated IDs.")

    return frame[[ID_COL, prediction_col]].copy()


def merge_predictions(
    lightgbm_path,
    pytorch_path,
    prediction_col,
    expected_rows,
):
    lightgbm = read_prediction(lightgbm_path, prediction_col, "LightGBM")
    pytorch = read_prediction(pytorch_path, prediction_col, "PyTorch")
    merged = lightgbm.merge(
        pytorch,
        on=ID_COL,
        how="inner",
        suffixes=("_lightgbm", "_pytorch"),
    )

    if len(merged) != expected_rows:
        raise ValueError(
            f"Prediction row mismatch: expected {expected_rows}, got {len(merged)}"
        )

    return merged


def probability_blend(lightgbm_prediction, pytorch_prediction, lightgbm_weight):
    pytorch_weight = 1.0 - lightgbm_weight
    return (
        lightgbm_weight * lightgbm_prediction
        + pytorch_weight * pytorch_prediction
    )


def rank_blend(lightgbm_prediction, pytorch_prediction, lightgbm_weight):
    pytorch_weight = 1.0 - lightgbm_weight
    return (
        lightgbm_weight * lightgbm_prediction.rank(pct=True)
        + pytorch_weight * pytorch_prediction.rank(pct=True)
    )


def scan_oof_blends():
    lightgbm_oof_path = next(
        (path for path in LIGHTGBM_OOF_CANDIDATES if path.exists()),
        None,
    )
    if lightgbm_oof_path is None:
        raise FileNotFoundError(
            f"LightGBM OOF file was not found: {LIGHTGBM_OOF_CANDIDATES}"
        )

    lightgbm_oof = pd.read_csv(lightgbm_oof_path)
    pytorch_oof = pd.read_csv(PYTORCH_OOF_PATH)
    required_cols = {ID_COL, TARGET_COL, OOF_PRED_COL}

    for frame, model_name in [
        (lightgbm_oof, "LightGBM OOF"),
        (pytorch_oof, "PyTorch OOF"),
    ]:
        missing_cols = sorted(required_cols - set(frame.columns))
        if missing_cols:
            raise ValueError(f"{model_name} file is missing columns: {missing_cols}")

    merged = lightgbm_oof[[ID_COL, TARGET_COL, OOF_PRED_COL]].merge(
        pytorch_oof[[ID_COL, TARGET_COL, OOF_PRED_COL]],
        on=[ID_COL, TARGET_COL],
        how="inner",
        suffixes=("_lightgbm", "_pytorch"),
    )
    if len(merged) != len(lightgbm_oof) or len(merged) != len(pytorch_oof):
        raise ValueError("OOF files do not contain the same train rows.")

    records = []
    for lightgbm_weight in np.arange(0.0, 1.001, 0.01):
        probability_prediction = probability_blend(
            merged[f"{OOF_PRED_COL}_lightgbm"],
            merged[f"{OOF_PRED_COL}_pytorch"],
            lightgbm_weight,
        )
        rank_prediction = rank_blend(
            merged[f"{OOF_PRED_COL}_lightgbm"],
            merged[f"{OOF_PRED_COL}_pytorch"],
            lightgbm_weight,
        )
        records.append(
            {
                "lightgbm_weight": round(float(lightgbm_weight), 2),
                "pytorch_weight": round(float(1.0 - lightgbm_weight), 2),
                "probability_blend_auc": roc_auc_score(
                    merged[TARGET_COL],
                    probability_prediction,
                ),
                "rank_blend_auc": roc_auc_score(
                    merged[TARGET_COL],
                    rank_prediction,
                ),
            }
        )

    metrics = pd.DataFrame(records)
    return metrics


def make_submission(test_predictions, prediction, output_path):
    submission = test_predictions[[ID_COL]].copy()
    submission[TARGET_COL] = np.clip(prediction, 0.0, 1.0)
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission: {output_path}")
    print(submission.head())


def main():
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    test_predictions = merge_predictions(
        LIGHTGBM_SUBMISSION_PATH,
        PYTORCH_SUBMISSION_PATH,
        TARGET_COL,
        expected_rows=188165,
    )
    metrics = scan_oof_blends()
    metrics.to_csv(DIAGNOSTIC_DIR / "blend_metrics.csv", index=False)

    best_probability = metrics.loc[metrics["probability_blend_auc"].idxmax()]
    best_rank = metrics.loc[metrics["rank_blend_auc"].idxmax()]
    selected = metrics.loc[
        np.isclose(metrics["lightgbm_weight"], LIGHTGBM_WEIGHT)
    ].iloc[0]

    print("=" * 60)
    print("OOF blend diagnostics")
    print("=" * 60)
    print(
        "Recommended rounded probability blend: "
        f"LightGBM={LIGHTGBM_WEIGHT:.2f}, PyTorch={PYTORCH_WEIGHT:.2f}, "
        f"AUC={selected['probability_blend_auc']:.6f}"
    )
    print(
        "Fine-grid best probability blend:       "
        f"LightGBM={best_probability['lightgbm_weight']:.2f}, "
        f"PyTorch={best_probability['pytorch_weight']:.2f}, "
        f"AUC={best_probability['probability_blend_auc']:.6f}"
    )
    print(
        "Fine-grid best rank blend:              "
        f"LightGBM={best_rank['lightgbm_weight']:.2f}, "
        f"PyTorch={best_rank['pytorch_weight']:.2f}, "
        f"AUC={best_rank['rank_blend_auc']:.6f}"
    )

    lightgbm_test = test_predictions[f"{TARGET_COL}_lightgbm"]
    pytorch_test = test_predictions[f"{TARGET_COL}_pytorch"]
    probability_prediction = probability_blend(
        lightgbm_test,
        pytorch_test,
        LIGHTGBM_WEIGHT,
    )
    rank_prediction = rank_blend(
        lightgbm_test,
        pytorch_test,
        LIGHTGBM_WEIGHT,
    )
    make_submission(
        test_predictions,
        probability_prediction,
        OUTPUT_PROBABILITY_PATH,
    )
    make_submission(
        test_predictions,
        rank_prediction,
        OUTPUT_RANK_PATH,
    )


if __name__ == "__main__":
    main()
