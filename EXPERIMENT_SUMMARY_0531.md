# F1 Pit Stop Prediction 実験概要

最終更新: 2026-06-01

## 現在地

目的は、次周にピットインする確率 `PitNextLap` を予測すること。評価指標は ROC AUC。

最終結果は、22nd Final BlendでPrivate Score `0.95433`、最終順位 `508` 位。

| 項目 | 値 |
|---|---:|
| Competition train rows | `439,140` |
| Test rows | `188,165` |
| External train rows | `101,371` |
| 現在の最高Public Score | `0.95392` |
| 現在の最高OOF AUC | `0.954707` |
| 最終Private Score | `0.95433` |
| 最終順位 | `508` |
| Team | `47` |
| Member | `Mizuki Nakai` |
| Total entries | `14` |

22nd本命と単純なlogit Blendを最終候補として選択した。どちらもPublic Scoreは
`0.95392`で、最終Private Scoreは `0.95433`まで上がった。

外部データとして `f1_strategy_dataset_v4.csv` を使用している。

## スコア推移

| Version | モデル | OOF AUC | Public Score | メモ |
|---|---|---:|---:|---|
| 1st | Raw LightGBM | - | `0.94196` | 最初の基準 |
| 4th | Raw LightGBM 3-seed ensemble | - | `0.94233` | seed平均が有効 |
| 5th | Tuned LightGBM 3-seed ensemble | `0.945251` | `0.94460` | 正則化を調整 |
| 7th | External-data LightGBM ensemble | `0.946474` | `0.94572` | 外部データを重み`0.65`で追加 |
| 9th | Lightweight PyTorch MLP | `0.941490` | 未提出 | 単体は弱いがBlend用に有効 |
| 10th | 7th + 9th Blend | `0.947876` | `0.94743` | `70%` LGBM + `30%` PyTorch |
| 11th | Advanced LightGBM | `0.948282` | 未提出 | カテゴリ交差、count、bin、fold-safe TE |
| 12th | 11th + 7th + 9th Blend | `0.949877` | 未提出 | 3モデルBlend |
| 13th | Advanced LightGBM 3-seed ensemble | `0.948884` | 未提出 | 11thをseed平均 |
| 13th Blend | 13th + 7th + 9th Blend | `0.950058` | `0.94916` | `60% + 25% + 15%` |
| 14th | Optuna-tuned Advanced LightGBM | `0.948730` | 未提出 | 1-seed。Blend候補として保持 |
| 15th | GPU PyTorch seed ensemble | `0.948970` | 未提出 | 通常MLP枝を大幅強化 |
| 17th | GPU RealMLP reference | `0.953732` | `0.95368` | 現在の単体best |
| 16th Blend | 17th + 7th + 15th Blend | `0.954127` | `0.95375` | `77% + 12% + 11%`、旧best |
| 18th | GPU RealMLP seed ensemble | `0.954225` | 未提出 | 17thを3-seed化。新しい単体anchor |
| 19th | GPU RealMLP 6-epoch seed ensemble | `0.954228` | 未提出 | 18thと実質同点。最終Blendの主anchor |
| 20th | Blend-oriented raw LightGBM | `0.951581` | 未提出 | `no_driver_ext65`を採択。RealMLP補完用 |
| 21st | GPU PyTorch residual complement | `0.950422` | 未提出 | 初回完走値。Kaggle session終了で成果物を失い、回収用rerun中 |
| 22nd | Final probability Blend | `0.954707` | `0.95392` | `55%` 19th + `20%` 20th + `19%` 18th + `6%` 15th |
| 22nd logit | Final logit-probability Blend | `0.954648` | `0.95392` | `73%` 19th + `27%` 20th。単純構成の保険候補 |

## 各モデルの役割

### LightGBM枝

13thは安定したLightGBM anchor。7thは13thより単体性能が低いが、予測差があるためBlendで残す価値がある。

14thではAdvanced LightGBMへOptunaを適用した。探索後の主な値:

```text
external_weight = 0.75
te_smoothing    = 20
num_leaves      = 31
learning_rate   = 0.0358579925
reg_lambda      = 4.8647145241
```

14th単体は13thを超えなかったが、13thより木が小さく、外部データを重く見る別系統としてBlend候補に残す。

LightGBMのGPU学習は、カテゴリ特徴のbin数がOpenCL GPU learnerの上限を超えたため使用できなかった。

```text
[LightGBM] [Fatal] bin size 501 cannot run on GPU
```

そのためAdvanced LightGBM系はCPUで回す。

### PyTorch枝

9thは軽量なEmbedding MLP。単体OOFは`0.941490`だったが、LightGBMと誤り方が異なり、10thと13th Blendで有効だった。

15thは9thを強化した通常MLP枝。

```text
2 seeds x 5 folds
最大20 epochs
mixed precision
early stopping
T4 x2 DataParallel対応
```

15th OOFは`0.948970`。LightGBM級まで伸びたため、17thとは異なるPyTorch補助モデルとして保持する。

### RealMLP枝

17thは `ps-s6-e5-realmlp-pytorch.ipynb` の強い構成をスクリプトへ移植したもの。

主な特徴:

```text
PBLD numerical embeddings
16 internal ensemble members
Residual blocks
parameter-group-specific learning rates
fold-safe TargetEncoder
external data
5 folds
```

参考NotebookのOOF `0.95373`をほぼ完全に再現し、17thでも`0.953732`となった。Public Scoreも`0.95368`で、後続RealMLP枝の基準となった。

## Blendの採択方法

これまでのBlendは、各モデルのOOF予測をIDで揃え、重みを総当たりしてROC AUCが最大になる組み合わせを採択している。

13th Blendの例:

```text
60% 13th Advanced LightGBM seed ensemble
25% 7th External-data LightGBM ensemble
15% 9th Lightweight PyTorch MLP

OOF AUC:     0.950058
Public Score: 0.94916
```

単体AUCだけではなく、モデル間の予測相関も重要。弱いモデルでも、anchorと異なる順位付けを持っていれば少量混ぜる価値がある。

今後は確率Blendだけでなく、Rank BlendとLogit Blendも比較する。

## 16th Blend探索

`16th_0531_realmlp_anchor_blend_search.py` を追加した。

16thは学習モデルではなく、ローカルで実行するBlend探索。17thを主anchorとして最低`55%`残し、既存モデルによる小さな補正を探索する。

候補:

```text
17th RealMLP anchor
13th Advanced LightGBM seed ensemble
7th External-data LightGBM ensemble
14th Optuna-tuned Advanced LightGBM
15th GPU PyTorch seed ensemble
9th Lightweight PyTorch MLP
```

自由度を上げすぎるとOOFへ過適合するため、最初は17thをanchorにして少量補正を探索する。

```text
17th + 13th
17th + 7th
17th + 14th
17th + 15th
17th + 13th + 7th
上位案への15th少量追加
```

探索対象:

```text
Probability Blend
Rank Remap Blend
Logit Probability Blend
Logit Rank Remap Blend
```

OOF最大値だけでなく、support weightを少し増減させた近傍AUCも保存する。OOF差が小さい場合は安定した候補を優先する。

ローカルで実行:

```bash
python 16th_0531_realmlp_anchor_blend_search.py
```

17th提出前に残り提出枠は5つだった。その後22nd本命、22nd logit候補、
21st追加版まで提出し、最終候補として22nd本命と22nd logit候補を選択した。

## 16th作成前にローカルへ戻すファイル

Kaggleから以下をローカルへ配置する。

```text
submission_15th_gpu_pytorch_seed_ensemble.csv
outputs/15th_gpu_pytorch_seed_ensemble/oof_predictions.csv

submission_17th_gpu_realmlp_reference.csv
outputs/17th_gpu_realmlp_reference/oof_predictions.csv
```

14thのsubmissionとOOFはすでにローカルにある。

## 主要ファイル

```text
13th_0531_advanced_lightgbm_seed_ensemble.py
13th_0531_seed_ensemble_three_model_blend.py
14th_0531_advanced_lightgbm_optuna.py
15th_0531_gpu_pytorch_seed_ensemble.py
17th_0531_gpu_realmlp_reference.py
16th_0531_realmlp_anchor_blend_search.py
18th_0531_gpu_realmlp_seed_ensemble.py
19th_0531_gpu_realmlp_6epoch_seed_ensemble.py
20th_0531_blend_optimized_lightgbm.py
21st_0531_gpu_pytorch_residual_complement.py
22nd_0531_final_oof_blend_search.py
```

## 18th RealMLP seed ensemble

17th の構造を変えず、seed を `42`, `2025`, `3407` の3本に増やした本命の安定化実験。
各seedで `5` folds、各fold内部で `16` ensemble members を学習する。

```bash
python 18th_0531_gpu_realmlp_seed_ensemble.py
```

出力:

```text
submission_18th_gpu_realmlp_seed_ensemble.csv
outputs/18th_gpu_realmlp_seed_ensemble/oof_predictions.csv
outputs/18th_gpu_realmlp_seed_ensemble/seed_oof_predictions.csv
outputs/18th_gpu_realmlp_seed_ensemble/fold_metrics.csv
outputs/18th_gpu_realmlp_seed_ensemble/seed_metrics.csv
```

P100 では現在の Kaggle PyTorch 環境とCUDA kernelの互換性エラーが発生した。
18th は `GPU T4 x2` で実行する。

18thの結果:

```text
OOF AUC: 0.954225
seed 42:   0.953732
seed 2025: 0.953692
seed 3407: 0.953749
```

全 `15/15` foldsで `best_epoch=4` だった。

## 19th RealMLP 6-epoch seed ensemble

18thの構造と3 seedsを維持したまま、最大epochを `4` から `6` へ延長する。
各foldのepoch別AUCも保存し、延長余地を確認する。

```bash
python 19th_0531_gpu_realmlp_6epoch_seed_ensemble.py
```

Kaggle `GPU T4 x2` で実行する。出力:

```text
submission_19th_gpu_realmlp_6epoch_seed_ensemble.csv
outputs/19th_gpu_realmlp_6epoch_seed_ensemble/oof_predictions.csv
outputs/19th_gpu_realmlp_6epoch_seed_ensemble/seed_oof_predictions.csv
outputs/19th_gpu_realmlp_6epoch_seed_ensemble/fold_metrics.csv
outputs/19th_gpu_realmlp_6epoch_seed_ensemble/epoch_metrics.csv
```

19thの結果:

```text
OOF AUC: 0.954228
seed 42:   0.953163
seed 2025: 0.953161
seed 3407: 0.953389
```

18thの `0.954225` と実質同点だが、22ndでは僅差で19thをanchorとして採択した。
ローカルへ回収した診断ファイルは旧フォルダ名の
`outputs/19th_pytorch_improve/` に保存されている。22ndはこのフォルダも
OOF候補として読む。

## 20th Blend-oriented LightGBM

LightGBM 側の補完モデル探索は 20th とする。
7th の raw LightGBM 系を起点に、RealMLPとのBlend AUCで候補を採択する。

```bash
python 20th_0531_blend_optimized_lightgbm.py
```

20th はローカルCPUで実行する。18thのOOFがローカルにあれば18thをanchorとして
優先し、なければ17thをanchorとして使う。出力:

```text
submission_20th_blend_optimized_lightgbm.csv
submission_20th_realmlp_lightgbm_probability_blend.csv
outputs/20th_blend_optimized_lightgbm/search_results.csv
outputs/20th_blend_optimized_lightgbm/final_blend_metrics.csv
```

20thの結果:

```text
Selected variant: no_driver_ext65
Standalone LightGBM OOF AUC: 0.951581
76% 18th RealMLP + 24% 20th LightGBM: 0.954571
```

## 21st GPU PyTorch residual complement

18th RealMLP と別系統の補完枝として、通常のcategorical embeddingを使うPyTorch MLPを強化する。
Residual MLP blocks、category dropout、戦略寄りの相互作用特徴、3-seed平均を使う。

```bash
python 21st_0531_gpu_pytorch_residual_complement.py
```

Kaggle `GPU T4 x2` で実行する。出力:

```text
submission_21st_gpu_pytorch_residual_complement.csv
outputs/21st_gpu_pytorch_residual_complement/oof_predictions.csv
outputs/21st_gpu_pytorch_residual_complement/seed_oof_predictions.csv
outputs/21st_gpu_pytorch_residual_complement/fold_metrics.csv
outputs/21st_gpu_pytorch_residual_complement/seed_metrics.csv
```

初回完走時の結果:

```text
OOF AUC: 0.950422
seed 42:   0.946236
seed 2025: 0.946315
seed 3407: 0.946428
```

初回の `/kaggle/working` 成果物はKaggle session終了により失われた。
21stを最終Blendへ加えるため、成果物回収用のrerunを行う。

## 22nd Final Blend

19thと18th RealMLP、20th LightGBM、21st PyTorchのOOFを比較し、
最終配合を探索する。既存の7th LightGBM、15th PyTorch、17th RealMLPも比較用に残す。
利用可能なRealMLPのうちOOFが最も高いモデルをanchorとして自動選択し、
最高OOF候補に加えて近傍weightでも崩れにくい保守候補を出力する。

```bash
python 22nd_0531_final_oof_blend_search.py
```

出力:

```text
submission_22nd_best_probability_blend.csv
submission_22nd_conservative_probability_blend.csv
submission_22nd_best_overall_blend.csv
outputs/22nd_final_oof_blend_search/finalists.csv
outputs/22nd_final_oof_blend_search/model_auc.csv
```

21stを含めない最初の22nd探索結果:

```text
Best probability blend:
55% 19th RealMLP
20% 20th LightGBM
19% 18th RealMLP
 6% 15th PyTorch
OOF AUC:      0.954707
Public Score: 0.95392

Simple logit-probability fallback:
73% 19th RealMLP
27% 20th LightGBM
OOF AUC:      0.954648
Public Score: 0.95392
```

両提出のPublic Scoreは同じ `0.95392`。21st成果物を加えた再探索版も提出したが、
Public Scoreは `0.95389`だったため、最終候補には採択しなかった。

最終結果:

```text
Final private score: 0.95433
Final rank:          508
Team:                47
Member:              Mizuki Nakai
Total entries:       14
```
