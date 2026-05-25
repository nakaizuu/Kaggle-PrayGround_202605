# Kaggle PrayGround 202605

LightGBM baseline for the F1 pit stop prediction competition.

## Results

- Evaluation metric: ROC AUC

| Version | File | Public Score | Notes |
|---|---|---:|---|
| 1st | `1st_0525_sub.py` | `0.94196` | Raw-feature LightGBM baseline. Rank: `1819` |
| 2nd | `2nd_0525_fe_sub.py` | `0.94130` | Added many row and group features. Score dropped slightly. |
| 3rd | `3rd_0525_lite_fe_sub.py` | `0.93930` | Light row-level feature engineering with 1st parameters. Score dropped further. |
| 4th | `4th_0525_raw_seed_ensemble.py` | `0.94233` | Raw-feature LightGBM with 3-seed ensemble. Current best. |

## Files

- `1st_0525_sub.py`: 5-fold LightGBM training and submission generation
- `2nd_0525_fe_sub.py`: LightGBM with basic feature engineering
- `3rd_0525_lite_fe_sub.py`: LightGBM with smaller row-level feature engineering
- `4th_0525_raw_seed_ensemble.py`: Raw-feature LightGBM seed ensemble
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

## Notes

The target column is `PitNextLap`. Since the competition is evaluated with ROC AUC between predicted probabilities and the observed target, the submission uses predicted probabilities rather than hard `0/1` labels.
