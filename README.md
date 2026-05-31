# Kaggle PrayGround 202605

LightGBM baseline for the F1 pit stop prediction competition.

## Results

- Evaluation metric: ROC AUC

| Version | File | Public Score | Notes |
|---|---|---:|---|
| 1st | `1st_0525_sub.py` | `0.94196` | Raw-feature LightGBM baseline. Rank: `1819` |
| 2nd | `2nd_0525_fe_sub.py` | `0.94130` | Added many row and group features. Score dropped slightly. |
| 3rd | `3rd_0525_lite_fe_sub.py` | `0.93930` | Light row-level feature engineering with 1st parameters. Score dropped further. |
| 4th | `4th_0525_raw_seed_ensemble.py` | `0.94233` | Raw-feature LightGBM with 3-seed ensemble. |
| 5th | `5th_0531_tuned_seed_ensemble.py` | `0.94460` | Raw-feature 3-seed ensemble with tuned LightGBM regularization. |
| 6th | `6th_0531_optuna_seed_ensemble.py` | TBD | Optuna search followed by a raw-feature 3-seed ensemble. |
| 7th | `7th_0531_external_data_seed_ensemble.py` | `0.94572` | 5th settings with public external training data weighted at `0.65`. Current best. |
| 8th | `8th_0531_external_fe_weighted_ensemble.py` | `0.94491` | Added small domain features and mild OOF AUC-weighted ensembling. The mean submission was tested, but OOF AUC dropped to `0.946003`. |

## Files

- `1st_0525_sub.py`: 5-fold LightGBM training and submission generation
- `2nd_0525_fe_sub.py`: LightGBM with basic feature engineering
- `3rd_0525_lite_fe_sub.py`: LightGBM with smaller row-level feature engineering
- `4th_0525_raw_seed_ensemble.py`: Raw-feature LightGBM seed ensemble
- `5th_0531_tuned_seed_ensemble.py`: Tuned raw-feature LightGBM seed ensemble with diagnostics
- `6th_0531_optuna_seed_ensemble.py`: Optuna-tuned raw-feature LightGBM seed ensemble with diagnostics
- `7th_0531_external_data_seed_ensemble.py`: Tuned raw-feature LightGBM seed ensemble with public external data
- `8th_0531_external_fe_weighted_ensemble.py`: External-data LightGBM with small domain features and mean/weighted submissions
- `requirements.txt`: Python package dependencies

The Kaggle data files are intentionally not tracked in Git.

## How to Run

Place the competition files in this directory:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Then run:

```bash
pip install -r requirements.txt
python 1st_0525_sub.py
```

The script creates:

```text
submission_lightgbm.csv
```

For the second feature-engineering version:

```bash
python 2nd_0525_fe_sub.py
```

The script creates:

```text
submission_lightgbm_fe.csv
```

For the third lite feature-engineering version:

```bash
python 3rd_0525_lite_fe_sub.py
```

The script creates:

```text
submission_lightgbm_lite_fe.csv
```

For the fourth raw-feature seed ensemble:

```bash
python 4th_0525_raw_seed_ensemble.py
```

The script creates:

```text
submission_lightgbm_raw_seed_ensemble.csv
```

For the fifth tuned raw-feature seed ensemble:

```bash
python 5th_0531_tuned_seed_ensemble.py
```

The script creates:

```text
submission_lightgbm_tuned_seed_ensemble.csv
```

Pre-submission diagnostics are written to:

```text
outputs/5th_tuned_seed_ensemble/
```

For the sixth Optuna-tuned seed ensemble:

```bash
python 6th_0531_optuna_seed_ensemble.py
```

The script first searches LightGBM parameters with 20 trials and 3-fold CV,
then trains a final 5-fold x 3-seed ensemble. It creates:

```text
submission_lightgbm_optuna_seed_ensemble.csv
```

Search and pre-submission diagnostics are written to:

```text
outputs/6th_optuna_seed_ensemble/
```

For the seventh external-data seed ensemble, also place the public external
dataset `f1_strategy_dataset_v4.csv` in this directory and run:

```bash
python 7th_0531_external_data_seed_ensemble.py
```

The script creates:

```text
submission_lightgbm_external_seed_ensemble.csv
```

Pre-submission diagnostics are written to:

```text
outputs/7th_external_data_seed_ensemble/
```

External dataset:

- [Formula 1 Strategy Dataset - Pit Stop Prediction](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction)

For the eighth external-data feature-engineering experiment:

```bash
python 8th_0531_external_fe_weighted_ensemble.py
```

The same trained models create both a plain mean submission and a mildly
OOF AUC-weighted submission:

```text
submission_lightgbm_external_fe_seed_mean.csv
submission_lightgbm_external_fe_seed_weighted.csv
```

Pre-submission diagnostics are written to:

```text
outputs/8th_external_fe_weighted_ensemble/
```

## Notes

The target column is `PitNextLap`. Since the competition is evaluated with ROC AUC between predicted probabilities and the observed target, the submission uses predicted probabilities rather than hard `0/1` labels.
