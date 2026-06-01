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
ADVANCED_LIGHTGBM_SUBMISSION_PATH = (
    DATA_DIR / "submission_11th_advanced_lightgbm_fold_te.csv"
)
PREVIOUS_LIGHTGBM_SUBMISSION_PATH = (
    DATA_DIR / "submission_lightgbm_external_seed_ensemble.csv"
)
PYTORCH_SUBMISSION_PATH = DATA_DIR / "submission_pytorch_tabular_mlp.csv"
ADVANCED_LIGHTGBM_OOF_CANDIDATES = [
    DATA_DIR / "outputs" / "11th_advanced_lightgbm" / "oof_predictions.csv",
    DATA_DIR
    / "outputs"
    / "11th_advanced_lightgbm_fold_te"
    / "oof_predictions.csv",
]
PREVIOUS_LIGHTGBM_OOF_CANDIDATES = [
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
OUTPUT_TWO_MODEL_PATH = (
    DATA_DIR / "submission_12th_advanced_lgbm_pytorch_probability_blend.csv"
)
OUTPUT_THREE_MODEL_PATH = (
    DATA_DIR / "submission_12th_three_model_probability_blend.csv"
)
DIAGNOSTIC_DIR = DATA_DIR / "outputs" / "12th_advanced_lgbm_three_model_blend"

ID_COL = "id"
TARGET_COL = "PitNextLap"
OOF_PRED_COL = "oof_prediction"
EXPECTED_TEST_ROWS = 188165
TWO_MODEL_STEP = 0.01
THREE_MODEL_STEP = 0.05


def find_existing_path(candidates, model_name):
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"{model_name} file was not found: {candidates}")
    return path


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


def read_oof(path, model_name):
    frame = pd.read_csv(path)
    required_cols = {ID_COL, TARGET_COL, OOF_PRED_COL}
    missing_cols = sorted(required_cols - set(frame.columns))
    if missing_cols:
        raise ValueError(f"{model_name} file is missing columns: {missing_cols}")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{model_name} file contains duplicated IDs.")

    return frame[[ID_COL, TARGET_COL, OOF_PRED_COL]].copy()


def merge_test_predictions():
    advanced = read_prediction(
        ADVANCED_LIGHTGBM_SUBMISSION_PATH,
        TARGET_COL,
        "Advanced LightGBM submission",
    ).rename(columns={TARGET_COL: "advanced_lightgbm"})
    previous = read_prediction(
        PREVIOUS_LIGHTGBM_SUBMISSION_PATH,
        TARGET_COL,
        "Previous LightGBM submission",
    ).rename(columns={TARGET_COL: "previous_lightgbm"})
    pytorch = read_prediction(
        PYTORCH_SUBMISSION_PATH,
        TARGET_COL,
        "PyTorch submission",
    ).rename(columns={TARGET_COL: "pytorch"})
    merged = advanced.merge(previous, on=ID_COL, how="inner").merge(
        pytorch,
        on=ID_COL,
        how="inner",
    )

    if len(merged) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Test prediction row mismatch: expected {EXPECTED_TEST_ROWS}, "
            f"got {len(merged)}"
        )

    return merged


def merge_oof_predictions():
    advanced_path = find_existing_path(
        ADVANCED_LIGHTGBM_OOF_CANDIDATES,
        "Advanced LightGBM OOF",
    )
    previous_path = find_existing_path(
        PREVIOUS_LIGHTGBM_OOF_CANDIDATES,
        "Previous LightGBM OOF",
    )
    advanced = read_oof(advanced_path, "Advanced LightGBM OOF").rename(
        columns={OOF_PRED_COL: "advanced_lightgbm"}
    )
    previous = read_oof(previous_path, "Previous LightGBM OOF").rename(
        columns={OOF_PRED_COL: "previous_lightgbm"}
    )
    pytorch = read_oof(PYTORCH_OOF_PATH, "PyTorch OOF").rename(
        columns={OOF_PRED_COL: "pytorch"}
    )
    merged = advanced.merge(
        previous,
        on=[ID_COL, TARGET_COL],
        how="inner",
    ).merge(
        pytorch,
        on=[ID_COL, TARGET_COL],
        how="inner",
    )

    if len(merged) != len(advanced):
        raise ValueError("OOF files do not contain the same train rows.")

    return merged


def blend_predictions(frame, advanced_weight, previous_weight, pytorch_weight):
    return (
        advanced_weight * frame["advanced_lightgbm"]
        + previous_weight * frame["previous_lightgbm"]
        + pytorch_weight * frame["pytorch"]
    )


def scan_two_model_blends(oof):
    records = []

    for advanced_weight in np.arange(0.0, 1.001, TWO_MODEL_STEP):
        pytorch_weight = 1.0 - advanced_weight
        prediction = blend_predictions(
            oof,
            advanced_weight,
            previous_weight=0.0,
            pytorch_weight=pytorch_weight,
        )
        records.append(
            {
                "advanced_lightgbm_weight": round(float(advanced_weight), 2),
                "pytorch_weight": round(float(pytorch_weight), 2),
                "auc": roc_auc_score(oof[TARGET_COL], prediction),
            }
        )

    return pd.DataFrame(records)


def scan_three_model_blends(oof):
    records = []
    unit_count = round(1.0 / THREE_MODEL_STEP)

    for advanced_units in range(unit_count + 1):
        for previous_units in range(unit_count - advanced_units + 1):
            pytorch_units = unit_count - advanced_units - previous_units
            advanced_weight = advanced_units * THREE_MODEL_STEP
            previous_weight = previous_units * THREE_MODEL_STEP
            pytorch_weight = pytorch_units * THREE_MODEL_STEP
            prediction = blend_predictions(
                oof,
                advanced_weight,
                previous_weight,
                pytorch_weight,
            )
            records.append(
                {
                    "advanced_lightgbm_weight": round(advanced_weight, 2),
                    "previous_lightgbm_weight": round(previous_weight, 2),
                    "pytorch_weight": round(pytorch_weight, 2),
                    "auc": roc_auc_score(oof[TARGET_COL], prediction),
                }
            )

    return pd.DataFrame(records).sort_values("auc", ascending=False)


def make_submission(test_predictions, prediction, output_path):
    submission = test_predictions[[ID_COL]].copy()
    submission[TARGET_COL] = np.clip(prediction, 0.0, 1.0)
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission: {output_path}")
    print(submission.head())


def main():
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    oof = merge_oof_predictions()
    test_predictions = merge_test_predictions()

    model_auc = pd.DataFrame(
        [
            {
                "model": model,
                "auc": roc_auc_score(oof[TARGET_COL], oof[model]),
            }
            for model in ["advanced_lightgbm", "previous_lightgbm", "pytorch"]
        ]
    )
    correlations = oof[
        ["advanced_lightgbm", "previous_lightgbm", "pytorch"]
    ].corr()
    two_model_metrics = scan_two_model_blends(oof)
    three_model_metrics = scan_three_model_blends(oof)
    best_two_model = two_model_metrics.loc[two_model_metrics["auc"].idxmax()]
    best_three_model = three_model_metrics.iloc[0]

    model_auc.to_csv(DIAGNOSTIC_DIR / "model_auc.csv", index=False)
    correlations.to_csv(DIAGNOSTIC_DIR / "correlations.csv")
    two_model_metrics.to_csv(DIAGNOSTIC_DIR / "two_model_blend_metrics.csv", index=False)
    three_model_metrics.to_csv(
        DIAGNOSTIC_DIR / "three_model_blend_metrics.csv",
        index=False,
    )

    print("=" * 60)
    print("OOF blend diagnostics")
    print("=" * 60)
    print("\nBase model AUC:")
    print(model_auc.to_string(index=False))
    print("\nOOF prediction correlations:")
    print(correlations.to_string())
    print(
        "\nFine-grid best 11th + PyTorch probability blend: "
        f"11th={best_two_model['advanced_lightgbm_weight']:.2f}, "
        f"PyTorch={best_two_model['pytorch_weight']:.2f}, "
        f"AUC={best_two_model['auc']:.6f}"
    )
    print("\nTop coarse-grid three-model probability blends:")
    print(three_model_metrics.head(10).to_string(index=False))

    two_model_prediction = blend_predictions(
        test_predictions,
        best_two_model["advanced_lightgbm_weight"],
        previous_weight=0.0,
        pytorch_weight=best_two_model["pytorch_weight"],
    )
    three_model_prediction = blend_predictions(
        test_predictions,
        best_three_model["advanced_lightgbm_weight"],
        best_three_model["previous_lightgbm_weight"],
        best_three_model["pytorch_weight"],
    )
    make_submission(
        test_predictions,
        two_model_prediction,
        OUTPUT_TWO_MODEL_PATH,
    )
    make_submission(
        test_predictions,
        three_model_prediction,
        OUTPUT_THREE_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
