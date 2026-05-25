# Kaggle PrayGround 202605

LightGBM baseline for the F1 pit stop prediction competition.

## Result

- Public score: `0.94196`
- Rank at first submission: `1819`
- Evaluation metric: ROC AUC

## Files

- `1st_0525_sub.py`: 5-fold LightGBM training and submission generation
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

## Notes

The target column is `PitNextLap`. Since the competition is evaluated with ROC AUC between predicted probabilities and the observed target, the submission uses predicted probabilities rather than hard `0/1` labels.
