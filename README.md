# Kaggle PrayGround 202605

LightGBM baseline for the F1 pit stop prediction competition.

Japanese experiment overview: [`EXPERIMENT_SUMMARY_0531.md`](EXPERIMENT_SUMMARY_0531.md)

## Results

- Evaluation metric: ROC AUC

### Final Result

| Item | Value |
|---|---:|
| Final leaderboard rank | `508` |
| Final private score | `0.95433` |
| Best public score | `0.95392` |
| Team | `47` |
| Member | `Mizuki Nakai` |
| Total entries | `14` |

The selected final submissions were the first 22nd best-probability blend and
the simpler 22nd logit-probability fallback. Both scored `0.95392` on the
public leaderboard. The final private score improved to `0.95433`.

| Version | File | Public Score | Notes |
|---|---|---:|---|
| 1st | `1st_0525_sub.py` | `0.94196` | Raw-feature LightGBM baseline. Rank: `1819` |
| 2nd | `2nd_0525_fe_sub.py` | `0.94130` | Added many row and group features. Score dropped slightly. |
| 3rd | `3rd_0525_lite_fe_sub.py` | `0.93930` | Light row-level feature engineering with 1st parameters. Score dropped further. |
| 4th | `4th_0525_raw_seed_ensemble.py` | `0.94233` | Raw-feature LightGBM with 3-seed ensemble. |
| 5th | `5th_0531_tuned_seed_ensemble.py` | `0.94460` | Raw-feature 3-seed ensemble with tuned LightGBM regularization. |
| 6th | `6th_0531_optuna_seed_ensemble.py` | TBD | Optuna search followed by a raw-feature 3-seed ensemble. |
| 7th | `7th_0531_external_data_seed_ensemble.py` | `0.94572` | 5th settings with public external training data weighted at `0.65`. |
| 8th | `8th_0531_external_fe_weighted_ensemble.py` | `0.94491` | Added small domain features and mild OOF AUC-weighted ensembling. The mean submission was tested, but OOF AUC dropped to `0.946003`. |
| 9th | `9th_0531_pytorch_tabular_mlp.py` | Not submitted | Lightweight RealMLP-inspired PyTorch model. OOF AUC: `0.941490`. Useful as an ensemble component. |
| 10th | `10th_0531_lightgbm_pytorch_blend.py` | `0.94743` | Rounded `70%` 7th LightGBM + `30%` 9th PyTorch probability blend. OOF AUC: `0.947876`. |
| 11th | `11th_0531_advanced_lightgbm_fold_te.py` | Not submitted | Advanced external-data LightGBM with category interactions, count encoding, bins, and fold-safe target encoding. OOF AUC: `0.948282`. |
| 12th | `12th_0531_advanced_lgbm_three_model_blend.py` | TBD | Rounded `55%` 11th LightGBM + `30%` 7th LightGBM + `15%` 9th PyTorch probability blend. OOF AUC: `0.949877`. |
| 13th | `13th_0531_advanced_lightgbm_seed_ensemble.py` | Not submitted | Three-seed ensemble of the 11th advanced LightGBM pipeline. OOF AUC: `0.948884`. |
| 13th blend | `13th_0531_seed_ensemble_three_model_blend.py` | `0.94916` | Rounded `60%` 13th LightGBM + `25%` 7th LightGBM + `15%` 9th PyTorch probability blend. OOF AUC: `0.950058`. |
| 14th | `14th_0531_advanced_lightgbm_optuna.py` | Not submitted | CPU Optuna search for the advanced LightGBM pipeline. OOF AUC: `0.948730`. |
| 15th | `15th_0531_gpu_pytorch_seed_ensemble.py` | Not submitted | Longer GPU PyTorch seed ensemble with wider embeddings, advanced categorical features, mixed precision, early stopping, and automatic T4 x2 use. OOF AUC: `0.948970`. |
| 16th | `16th_0531_realmlp_anchor_blend_search.py` | `0.95375` | Former best: `77%` 17th RealMLP + `12%` 7th LightGBM + `11%` 15th PyTorch. OOF AUC: `0.954127`. |
| 17th | `17th_0531_gpu_realmlp_reference.py` | `0.95368` | Reference-style GPU RealMLP with PBLD numerical embeddings and `16` internal ensemble members. OOF AUC: `0.953732`. |
| 18th | `18th_0531_gpu_realmlp_seed_ensemble.py` | Not submitted | Three-seed stabilization of the 17th RealMLP. OOF AUC: `0.954225`. |
| 19th | `19th_0531_gpu_realmlp_6epoch_seed_ensemble.py` | Not submitted | Six-epoch RealMLP follow-up. OOF AUC: `0.954228`, effectively tied with 18th but selected as the final anchor. |
| 20th | `20th_0531_blend_optimized_lightgbm.py` | Not submitted | Selected `no_driver_ext65`: standalone OOF `0.951581`; `76%` 18th RealMLP + `24%` 20th LightGBM reaches OOF `0.954571`. |
| 21st | `21st_0531_gpu_pytorch_residual_complement.py` | Not submitted | First completed run reached OOF `0.950422`, but its Kaggle working-directory artifacts were lost after the session ended. A recovery rerun is pending. |
| 22nd | `22nd_0531_final_oof_blend_search.py` | `0.95392` | Final selected candidates. Best probability blend OOF: `0.954707` from `55%` 19th + `20%` 20th + `19%` 18th + `6%` 15th. The simpler `73%` 19th + `27%` 20th logit blend also scored `0.95392` Public. Final private score: `0.95433`; rank: `508`. |

## Files

- `1st_0525_sub.py`: 5-fold LightGBM training and submission generation
- `2nd_0525_fe_sub.py`: LightGBM with basic feature engineering
- `3rd_0525_lite_fe_sub.py`: LightGBM with smaller row-level feature engineering
- `4th_0525_raw_seed_ensemble.py`: Raw-feature LightGBM seed ensemble
- `5th_0531_tuned_seed_ensemble.py`: Tuned raw-feature LightGBM seed ensemble with diagnostics
- `6th_0531_optuna_seed_ensemble.py`: Optuna-tuned raw-feature LightGBM seed ensemble with diagnostics
- `7th_0531_external_data_seed_ensemble.py`: Tuned raw-feature LightGBM seed ensemble with public external data
- `8th_0531_external_fe_weighted_ensemble.py`: External-data LightGBM with small domain features and mean/weighted submissions
- `9th_0531_pytorch_tabular_mlp.py`: RealMLP-inspired PyTorch tabular model with embeddings and fold-safe target encoding
- `10th_0531_lightgbm_pytorch_blend.py`: Fast local blend of the 7th LightGBM and 9th PyTorch predictions
- `11th_0531_advanced_lightgbm_fold_te.py`: Advanced external-data LightGBM with fold-safe target encoding
- `12th_0531_advanced_lgbm_three_model_blend.py`: Fast local OOF blend scan for the 11th, 7th, and 9th models
- `13th_0531_advanced_lightgbm_seed_ensemble.py`: Three-seed ensemble of the advanced LightGBM pipeline
- `13th_0531_seed_ensemble_three_model_blend.py`: Fast local OOF blend scan for the 13th, 7th, and 9th models
- `14th_0531_advanced_lightgbm_optuna.py`: CPU Optuna search and final 5-fold verification for the advanced LightGBM pipeline
- `18th_0531_gpu_realmlp_seed_ensemble.py`: Three-seed stabilization of the reference RealMLP
- `19th_0531_gpu_realmlp_6epoch_seed_ensemble.py`: Six-epoch follow-up to test whether the RealMLP anchor still has room to improve
- `20th_0531_blend_optimized_lightgbm.py`: Blend-oriented raw LightGBM CPU search and final three-seed training
- `21st_0531_gpu_pytorch_residual_complement.py`: Diverse residual embedding MLP complement for the RealMLP anchor
- `22nd_0531_final_oof_blend_search.py`: Final local OOF blend search with automatic RealMLP anchor selection and conservative candidates
- `15th_0531_gpu_pytorch_seed_ensemble.py`: Longer GPU PyTorch seed ensemble for a stronger diverse blend component
- `16th_0531_realmlp_anchor_blend_search.py`: Local OOF blend search centered on the 17th RealMLP anchor
- `17th_0531_gpu_realmlp_reference.py`: Reference-style GPU RealMLP with PBLD numerical embeddings and a 16-member internal ensemble
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

For the ninth RealMLP-inspired PyTorch experiment:

```bash
python 9th_0531_pytorch_tabular_mlp.py
```

The script uses the public external dataset, fold-safe target encoding, and
categorical embeddings. A CUDA GPU is strongly recommended. It creates:

```text
submission_pytorch_tabular_mlp.csv
```

Pre-submission diagnostics and LightGBM blend candidates are written to:

```text
outputs/9th_pytorch_tabular_mlp/
```

For the tenth LightGBM and PyTorch blend:

```bash
python 10th_0531_lightgbm_pytorch_blend.py
```

The script scans OOF blend weights and creates:

```text
submission_10th_lgbm70_pytorch30_probability_blend.csv
submission_10th_lgbm70_pytorch30_rank_blend.csv
```

The rounded `70%` LightGBM and `30%` PyTorch probability blend is the primary
submission candidate. Blend diagnostics are written to:

```text
outputs/10th_lgbm_pytorch_blend/
```

For the eleventh advanced LightGBM experiment:

```bash
python 11th_0531_advanced_lightgbm_fold_te.py
```

The script supports both the local project directory and the Kaggle input
dataset at `/kaggle/input/datasets/mizukinakaizuuu/input-4data`. On Kaggle, the
submission and diagnostics are written under `/kaggle/working`. It creates:

```text
submission_11th_advanced_lightgbm_fold_te.csv
```

Pre-submission diagnostics are written to:

```text
outputs/11th_advanced_lightgbm_fold_te/
```

For the twelfth local blend scan:

```bash
python 12th_0531_advanced_lgbm_three_model_blend.py
```

The script compares the 11th advanced LightGBM, 7th LightGBM, and 9th
PyTorch OOF predictions. It creates two probability-blend candidates:

```text
submission_12th_advanced_lgbm_pytorch_probability_blend.csv
submission_12th_three_model_probability_blend.csv
```

Blend diagnostics are written to:

```text
outputs/12th_advanced_lgbm_three_model_blend/
```

For the thirteenth advanced LightGBM seed ensemble:

```bash
python 13th_0531_advanced_lightgbm_seed_ensemble.py
```

The script trains the 11th advanced pipeline with three seeds and creates:

```text
submission_13th_advanced_lightgbm_seed_ensemble.csv
```

Seed-level OOF predictions and pre-submission diagnostics are written to:

```text
outputs/13th_advanced_lightgbm_seed_ensemble/
```

After copying the thirteenth submission and diagnostics to the local project,
scan updated blend weights with:

```bash
python 13th_0531_seed_ensemble_three_model_blend.py
```

The script creates:

```text
submission_13th_seed_lgbm_pytorch_probability_blend.csv
submission_13th_seed_three_model_probability_blend.csv
```

For the fourteenth advanced LightGBM Optuna search:

```bash
python 14th_0531_advanced_lightgbm_optuna.py
```

The script runs a CPU `15`-trial x `3`-fold Optuna search, then verifies the
best parameters with a final `5`-fold model. It creates:

```text
submission_14th_advanced_lightgbm_optuna.csv
```

The best parameters, trial history, OOF predictions, and final diagnostics are
written to:

```text
outputs/14th_advanced_lightgbm_optuna/
```

The Optuna study is persisted after each trial in:

```text
outputs/14th_advanced_lightgbm_optuna/optuna_study.db
```

This makes the local CPU search resumable. Run the same command again after an
interruption to continue the remaining trials.

For the fifteenth GPU PyTorch seed ensemble:

```bash
python 15th_0531_gpu_pytorch_seed_ensemble.py
```

Run this one on Kaggle with the `GPU T4 x2` accelerator. The script uses mixed
precision and automatically wraps the model with `DataParallel` when two GPUs
are visible. It trains two seeds with `5` folds, up to `20` epochs per fold,
and restores the best epoch after early stopping. It creates:

```text
submission_15th_gpu_pytorch_seed_ensemble.csv
```

Seed-level OOF predictions and pre-submission diagnostics are written to:

```text
outputs/15th_gpu_pytorch_seed_ensemble/
```

For the seventeenth reference-style GPU RealMLP:

```bash
python 17th_0531_gpu_realmlp_reference.py
```

This ports the strong public reference notebook into the local experiment
layout. It uses PBLD numerical embeddings and `16` internal ensemble members.
Run it on Kaggle with the `GPU T4 x2` accelerator. This reproduction-oriented
version intentionally uses GPU `0` only. It creates:

```text
submission_17th_gpu_realmlp_reference.csv
```

OOF predictions, fold metrics, configuration, and model checkpoints are
written to:

```text
outputs/17th_gpu_realmlp_reference/
```

For the eighteenth GPU RealMLP seed ensemble:

```bash
python 18th_0531_gpu_realmlp_seed_ensemble.py
```

This keeps the seventeenth RealMLP architecture and trains three seeds with
`5` folds each. Each fold still contains `16` internal ensemble members. Run
it on Kaggle with the `GPU T4 x2` accelerator. The current Kaggle PyTorch
environment may reject the older `GPU P100` with a CUDA kernel compatibility
error. It creates:

```text
submission_18th_gpu_realmlp_seed_ensemble.csv
```

The averaged OOF predictions, per-seed OOF predictions, fold metrics, seed
metrics, configuration, and model checkpoints are written to:

```text
outputs/18th_gpu_realmlp_seed_ensemble/
```

For the nineteenth six-epoch GPU RealMLP:

```bash
python 19th_0531_gpu_realmlp_6epoch_seed_ensemble.py
```

The eighteenth run improved the standalone OOF AUC to `0.954225`, and every
fold selected the final epoch `4/4`. This follow-up keeps the architecture,
three seeds, five folds, and internal `16`-member ensemble unchanged while
extending training to `6` epochs. Run it on Kaggle with `GPU T4 x2`. It
creates:

```text
submission_19th_gpu_realmlp_6epoch_seed_ensemble.csv
```

Diagnostics, including per-epoch validation AUC values, are written to:

```text
outputs/19th_gpu_realmlp_6epoch_seed_ensemble/
```

The completed run reached OOF AUC `0.954228`. The locally recovered copy of
these diagnostics currently uses the older folder name:

```text
outputs/19th_pytorch_improve/
```

For the twentieth local blend-oriented LightGBM search:

```bash
python 20th_0531_blend_optimized_lightgbm.py
```

This keeps the raw seventh LightGBM family as a deliberately different model
from RealMLP. It compares compact feature sets, external-data weights, and
regularized parameter profiles using `3`-fold CPU CV. Candidate selection is
based on RealMLP + LightGBM OOF blend AUC rather than standalone LightGBM AUC.
The selected model is then trained with three seeds and `5` folds. It creates:

```text
submission_20th_blend_optimized_lightgbm.csv
submission_20th_realmlp_lightgbm_probability_blend.csv
```

Diagnostics are written to:

```text
outputs/20th_blend_optimized_lightgbm/
```

The selected `no_driver_ext65` branch reached standalone OOF AUC `0.951581`.
Its probability blend with the eighteenth RealMLP peaked at OOF AUC `0.954571`
with `76%` RealMLP and `24%` LightGBM.

For the twenty-first GPU PyTorch residual complement:

```bash
python 21st_0531_gpu_pytorch_residual_complement.py
```

Run this on Kaggle with the `GPU T4 x2` accelerator. It keeps a standard
categorical-embedding PyTorch branch deliberately separate from RealMLP,
then adds residual MLP blocks, category dropout, strategy interactions, and
three-seed stabilization. When the nineteenth or eighteenth OOF file is
available as a Kaggle dataset, it also writes optional RealMLP blend diagnostics.
It creates:

```text
submission_21st_gpu_pytorch_residual_complement.csv
```

Diagnostics are written to:

```text
outputs/21st_gpu_pytorch_residual_complement/
```

The first completed Kaggle run reached OOF AUC `0.950422`. Its files were lost
when the Kaggle session ended, so a recovery rerun is needed before this branch
can be included in the final local blend search.

For the twenty-second final local OOF blend search:

```bash
python 22nd_0531_final_oof_blend_search.py
```

Run this after copying the nineteenth and twenty-first Kaggle outputs locally.
It automatically chooses the strongest available RealMLP anchor, compares the
new twentieth and twenty-first branches with the useful older seventh and
fifteenth complements, and writes both best-OOF and conservative candidates.
Diagnostics are written to:

```text
outputs/22nd_final_oof_blend_search/
```

The first final search, without the twenty-first branch, selected:

```text
55% 19th RealMLP + 20% 20th LightGBM + 19% 18th RealMLP + 6% 15th PyTorch
OOF AUC:      0.954707
Public Score: 0.95392
```

The simpler logit-probability fallback was also submitted:

```text
73% 19th RealMLP + 27% 20th LightGBM
OOF AUC:      0.954648
Public Score: 0.95392
```

For the sixteenth local RealMLP-anchor blend search:

```bash
python 16th_0531_realmlp_anchor_blend_search.py
```

The script automatically recognizes both the canonical Kaggle diagnostic
folder names and the shorter local copy names. It evaluates anchored
probability, rank-remap, logit-probability, and logit-rank-remap blends. It
writes candidate submissions and detailed OOF diagnostics to:

```text
outputs/16th_realmlp_anchor_blend_search/
```

## Notes

The target column is `PitNextLap`. Since the competition is evaluated with ROC AUC between predicted probabilities and the observed target, the submission uses predicted probabilities rather than hard `0/1` labels.
