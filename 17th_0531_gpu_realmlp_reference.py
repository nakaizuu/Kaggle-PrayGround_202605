from pathlib import Path
import gc
import json
import math
import random
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
    from sklearn.utils.class_weight import compute_class_weight
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Required package is missing. Install dependencies with:\n"
        "pip install torch scikit-learn pandas numpy"
    ) from exc


warnings.filterwarnings("ignore")

LOCAL_DATA_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)
KAGGLE_DATA_DIR = Path("/kaggle/input/datasets/mizukinakaizuuu/input-4data")

if (KAGGLE_DATA_DIR / "train.csv").exists():
    INPUT_DIR = KAGGLE_DATA_DIR
    WORK_DIR = Path("/kaggle/working")
else:
    INPUT_DIR = LOCAL_DATA_DIR
    WORK_DIR = LOCAL_DATA_DIR

TRAIN_PATH = INPUT_DIR / "train.csv"
TEST_PATH = INPUT_DIR / "test.csv"
SAMPLE_PATH = INPUT_DIR / "sample_submission.csv"
EXTERNAL_PATH = INPUT_DIR / "f1_strategy_dataset_v4.csv"
OUTPUT_PATH = WORK_DIR / "submission_17th_gpu_realmlp_reference.csv"
DIAGNOSTIC_DIR = WORK_DIR / "outputs" / "17th_gpu_realmlp_reference"
CHECKPOINT_DIR = DIAGNOSTIC_DIR / "checkpoints"

ID_COL = "id"
TARGET_COL = "PitNextLap"
N_SPLITS = 5
SEED = 42
USE_TARGET_ENCODING = True
REQUIRE_CUDA = True

CONFIG = {
    "n_ens": 16,
    "embed_dim": 6,
    "onehot_thresh": 4,
    "hidden_dims": [256, 256, 256],
    "dropout": 0.05,
    "p_drop_sched": "expm4t",
    "activation": nn.SiLU,
    "add_front_scale": True,
    "pbld_hidden_dim": 20,
    "pbld_out_dim": 5,
    "pbld_freq_scale": 5.0,
    "pbld_activation": nn.PReLU,
    "pbld_lr_factor": 0.093,
    "lr": 0.008,
    "mom": 0.9,
    "sq_mom": 0.98,
    "lr_sched": "flat_cos",
    "flat_ratio": 0.3,
    "first_layer_lr_factor": 1.0,
    "first_layer_wd_factor": 0.1,
    "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1,
    "weight_decay": 0.005,
    "wd_scale_mult": 0.1,
    "wd_bias_mult": 0.5,
    "grad_clip": 1.0,
    "ls_eps": 0.04,
    "ls_eps_sched": "cos",
    "tfms": ["median_center", "robust_scale", "smooth_clip"],
    "epochs": 4,
    "train_bs": 256,
    "eval_bs": 10240,
    "verbosity": 2,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "device": "cuda",
    "random_state": SEED,
}


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class FeatureEngineer:
    def __init__(self, categorical_cols, numeric_cols):
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.category_map = {}
        self.important_combos = [
            ("Race", "Compound"),
            ("Race", "Year"),
        ]

    def transform(self, frame, fit=False):
        frame = frame.copy()
        frame["_LapNumber_/_RaceProgress"] = (
            frame["LapNumber"] / (frame["RaceProgress"] + 1e-6)
        ).astype("float32")
        frame["_TyreLife_/_LapNumber"] = (
            frame["TyreLife"] / frame["LapNumber"].clip(lower=1)
        ).astype("float32")
        frame["_LapTime (s)_*_Cumulative_Degradation"] = (
            frame["LapTime (s)"] * frame["Cumulative_Degradation"]
        ).astype("float32")
        frame["_LapTime (s)_*_Cumulative_Degradation_abs"] = (
            frame["LapTime (s)"] * frame["Cumulative_Degradation"].abs()
        ).astype("float32")
        frame["_LapTime (s)_/_Cumulative_Degradation_abs"] = (
            frame["LapTime (s)"]
            / (frame["Cumulative_Degradation"].abs() + 1e-6)
        ).astype("float32")

        for col in self.categorical_cols:
            if fit:
                codes, uniques = frame[col].factorize()
                self.category_map[col] = uniques
            else:
                uniques = self.category_map[col]
                code_map = {cat: index for index, cat in enumerate(uniques)}
                codes = frame[col].map(code_map).fillna(-1).astype("int32")
            frame[col] = codes
            frame[col] = frame[col].astype("category")

        categorized_numeric_cols = self.numeric_cols + [
            "_LapNumber_/_RaceProgress",
            "_TyreLife_/_LapNumber",
        ]
        for col in categorized_numeric_cols:
            cat_name = f"{col}_cat_" if col in self.numeric_cols else f"{col[1:]}_cat_"
            if fit:
                codes, uniques = np.floor(frame[col]).factorize()
                self.category_map[col] = uniques
            else:
                uniques = self.category_map[col]
                code_map = {cat: index for index, cat in enumerate(uniques)}
                codes = (
                    np.floor(frame[col])
                    .map(code_map)
                    .fillna(-1)
                    .astype("int32")
                )
            frame[cat_name] = codes
            frame[cat_name] = frame[cat_name].astype("category")

        for col in self.categorical_cols + ["Year_cat_", "PitStop_cat_"]:
            count_name = f"_{col}_count" if col in self.categorical_cols else f"_{col[:-1]}_count"
            if fit:
                count_map = frame[col].value_counts()
                self.category_map[count_name] = count_map
            else:
                count_map = self.category_map[count_name]
            frame[count_name] = (
                frame[col].astype(object).map(count_map).fillna(0).astype("int32")
            )

        bin_config = {"RaceProgress": [200], "LapTime (s)": [7]}
        for col, bins_list in bin_config.items():
            for n_bins in bins_list:
                bin_name = f"{col}_{n_bins}_quantile_bin_"
                if fit:
                    discretizer = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="ordinal",
                        strategy="quantile",
                        subsample=None,
                    )
                    binned = (
                        discretizer.fit_transform(frame[[col]])
                        .ravel()
                        .astype("int32")
                    )
                    self.category_map[bin_name] = discretizer
                else:
                    discretizer = self.category_map[bin_name]
                    binned = (
                        discretizer.transform(frame[[col]])
                        .ravel()
                        .astype("int32")
                    )
                frame[bin_name] = binned
                frame[bin_name] = frame[bin_name].astype("category")

        combo_names = []
        for cols in self.important_combos:
            combo_name = "_".join(cols) + "_"
            combo_names.append(combo_name)
            combo_series = frame[cols[0]].astype(str)
            for col in cols[1:]:
                combo_series = combo_series + "_" + frame[col].astype(str)
            if fit:
                codes, uniques = pd.factorize(combo_series, sort=False)
                self.category_map[combo_name] = uniques
            else:
                uniques = self.category_map[combo_name]
                code_map = {cat: index for index, cat in enumerate(uniques)}
                codes = combo_series.map(code_map).fillna(-1).astype("int32")
            frame[combo_name] = codes
            frame[combo_name] = frame[combo_name].astype("category")

        new_categorical_cols = [col for col in frame.columns if col.endswith("_")]
        new_numeric_cols = [col for col in frame.columns if col.startswith("_")]
        return frame, new_categorical_cols, new_numeric_cols, combo_names


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [
            tfm
            for tfm in tfms
            if tfm in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")
        ]

    def fit(self, X, y=None):
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.median(X, axis=0)
            q_diff = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
            zero_idx = q_diff == 0.0
            q_diff[zero_idx] = 0.5 * (
                X.max(axis=0)[zero_idx] - X.min(axis=0)[zero_idx]
            )
            self._iqr_factors = 1.0 / (q_diff + 1e-30)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = X.copy().astype(np.float32)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = cat_dims
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []

        for index, dim in enumerate(cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(index)
            else:
                self.embed_layers.append(
                    nn.ModuleList(
                        [nn.Embedding(dim, embed_dim) for _ in range(n_ens)]
                    )
                )
                self._embed_feature_indices.append(index)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []

        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[index] for index in self.onehot_features]
            encoded = torch.zeros(
                batch_size,
                n_ens,
                sum(onehot_dims),
                device=x.device,
            )
            start = 0
            for index, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, index : index + 1].long()
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)

        for embedding_list, feature_index in zip(
            self.embed_layers,
            self._embed_feature_indices,
        ):
            feature_embeddings = []
            for model_index in range(self.n_ens):
                indices = x[:, model_index, feature_index : feature_index + 1].long()
                feature_embeddings.append(embedding_list[model_index](indices))
            features.append(torch.cat(feature_embeddings, dim=1))

        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = (
            nn.Parameter(torch.randn(n_ens, out_features))
            if bias
            else None
        )

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(
            self.in_features
        )
        if self.bias is not None:
            x = x + self.bias
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_ens, dim, dropout, activation=nn.SiLU):
        super().__init__()
        self.linear = NTPLinear(
            n_ens=n_ens,
            in_features=dim,
            out_features=dim,
        )
        self.act = activation()
        self.drop = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.ones(n_ens, dim) * 0.1)

    def forward(self, x):
        residual = x
        x = self.linear(x)
        x = self.act(x)
        x = self.drop(x)
        return residual + x * self.res_scale.unsqueeze(0)


class PBLDEmbedding(nn.Module):
    def __init__(
        self,
        n_ens,
        n_features,
        hidden_dim=16,
        out_dim=4,
        freq_scale=0.1,
        activation=nn.GELU,
    ):
        super().__init__()
        self.n_ens = n_ens
        self.n_features = n_features
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.empty(n_ens, n_features, hidden_dim))
        nn.init.normal_(
            self.w1,
            mean=0.0,
            std=freq_scale / math.sqrt(hidden_dim),
        )
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(
            torch.randn(n_ens, n_features, hidden_dim, out_dim - 1)
            / math.sqrt(hidden_dim)
        )
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(
            2
            * math.pi
            * (
                x.unsqueeze(-1) * self.w1.unsqueeze(0)
                + self.b1.unsqueeze(0)
            )
        )
        transformed = self.act(
            torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2)
            + self.b2.unsqueeze(0)
        )
        return torch.cat([x.unsqueeze(-1), transformed], dim=-1).flatten(
            start_dim=2
        )


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        embed_dim = cfg["embed_dim"]
        self.n_ens = n_ens

        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens,
            cat_dims=cat_dims,
            embed_dim=embed_dim,
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens,
            n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"],
            out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"],
            activation=cfg["pbld_activation"],
        )

        num_embedding_dim = n_numerical * cfg["pbld_out_dim"]
        cat_embedding_dim = sum(
            dim if dim <= cfg["onehot_thresh"] else embed_dim
            for dim in cat_dims
        )
        total_dim = num_embedding_dim + cat_embedding_dim
        hidden_dims = cfg["hidden_dims"]
        activation = cfg["activation"]
        self._dropout_modules = []

        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))

        input_dim = total_dim
        self.first_linear = NTPLinear(
            n_ens=n_ens,
            in_features=input_dim,
            out_features=hidden_dims[0],
        )
        layers.extend([self.first_linear, activation()])
        input_dim = hidden_dims[0]

        for hidden_dim in hidden_dims[1:]:
            if input_dim != hidden_dim:
                layers.extend(
                    [
                        NTPLinear(
                            n_ens=n_ens,
                            in_features=input_dim,
                            out_features=hidden_dim,
                        ),
                        activation(),
                    ]
                )
                input_dim = hidden_dim

            block = ResidualBlock(
                n_ens=n_ens,
                dim=hidden_dim,
                dropout=cfg["dropout"],
                activation=activation,
            )
            self._dropout_modules.append(block.drop)
            layers.append(block)

        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(
            n_ens=n_ens,
            in_features=input_dim,
            out_features=output_dim,
        )
        with torch.no_grad():
            self.output_layer.weight.mul_(0.1)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_num = self.num_embed(x_num)
        x_cat = self.cate(x_cat)
        combined = torch.cat([x_num, x_cat], dim=2)
        return self.output_layer(self.hidden(combined))


def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init_value
    if sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    if sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        adjusted = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * adjusted) + 1) / 2
    if sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        adjusted = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - adjusted)
    if sched == "sqrt_cos":
        return init_value * math.sqrt(
            (math.cos(math.pi * progress) + 1) / 2
        )
    if sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    raise ValueError(f"Unknown schedule: '{sched}'")


def get_parameter_groups(model, params):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_params = []
    pbld_params = []
    first_weight_params = []
    other_weight_params = []
    bias_params = []

    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_params.append(param)
        elif "scale" in name:
            scale_params.append(param)
        elif id(param) == first_linear_weight_id:
            first_weight_params.append(param)
        elif "bias" in name:
            bias_params.append(param)
        else:
            other_weight_params.append(param)

    lr = params["lr"]
    weight_decay = params["weight_decay"]
    return [
        {
            "params": scale_params,
            "lr": lr * params["lr_scale_mult"],
            "weight_decay": weight_decay * params["wd_scale_mult"],
        },
        {
            "params": pbld_params,
            "lr": lr * params["pbld_lr_factor"],
            "weight_decay": weight_decay,
        },
        {
            "params": first_weight_params,
            "lr": lr * params["first_layer_lr_factor"],
            "weight_decay": weight_decay * params["first_layer_wd_factor"],
        },
        {
            "params": other_weight_params,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": bias_params,
            "lr": lr * params["lr_bias_mult"],
            "weight_decay": weight_decay * params["wd_bias_mult"],
        },
    ]


def binary_bce_loss(y_true, logits, label_smoothing=0.0, pos_weight=None):
    if label_smoothing > 0.0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if pos_weight is None:
        loss = (1.0 - y_true) * logits + F.softplus(-logits)
    else:
        loss = (
            (1.0 - y_true) * logits
            + (1.0 + (pos_weight - 1.0) * y_true) * F.softplus(-logits)
        )
    return loss.mean()


class RealMLPClassifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = {**CONFIG, **kwargs}

    def fit(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_cols=None,
        checkpoint_path="realmlp_checkpoint.pth",
    ):
        params = self.params
        device = torch.device(params["device"])
        categorical_cols = categorical_cols or []
        numeric_cols = [
            col for col in X_train.columns if col not in categorical_cols
        ]

        X_train_numeric = X_train[numeric_cols].values.astype(np.float32)
        X_valid_numeric = X_valid[numeric_cols].values.astype(np.float32)
        X_train_categorical = X_train[categorical_cols].values.astype(np.int64)
        X_valid_categorical = X_valid[categorical_cols].values.astype(np.int64)
        y_train_array = np.asarray(y_train)
        y_valid_array = np.asarray(y_valid)

        self.preprocessor_ = NumericalPreprocessor(params["tfms"])
        self.preprocessor_.fit(X_train_numeric)
        X_train_numeric = self.preprocessor_.transform(X_train_numeric)
        X_valid_numeric = self.preprocessor_.transform(X_valid_numeric)
        self.categorical_cols_ = categorical_cols
        self.numeric_cols_ = numeric_cols

        if categorical_cols:
            categorical_dims = (
                np.concatenate(
                    [X_train_categorical, X_valid_categorical],
                    axis=0,
                ).max(axis=0)
                + 1
            ).tolist()
        else:
            categorical_dims = []
        self.categorical_dims_ = categorical_dims

        if categorical_dims:
            categorical_max = np.array(categorical_dims) - 1
            X_train_categorical = np.clip(
                X_train_categorical,
                0,
                categorical_max,
            )
            X_valid_categorical = np.clip(
                X_valid_categorical,
                0,
                categorical_max,
            )

        classes = np.unique(y_train_array)
        self.classes_ = classes
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train_array,
        )
        pos_weight = torch.tensor(
            class_weights[1],
            dtype=torch.float32,
            device=device,
        )

        self.model_ = RealMLP(
            output_dim=1,
            cat_dims=categorical_dims,
            n_numerical=X_train_numeric.shape[1],
            cfg=params,
        ).to(device)
        parameter_groups = get_parameter_groups(self.model_, params)
        for group in parameter_groups:
            group["lr_base"] = group["lr"]
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=(params["mom"], params["sq_mom"]),
        )

        X_train_numeric_tensor = torch.as_tensor(
            X_train_numeric,
            dtype=torch.float32,
            device=device,
        )
        X_train_categorical_tensor = torch.as_tensor(
            X_train_categorical,
            dtype=torch.long,
            device=device,
        )
        y_train_tensor = torch.as_tensor(
            y_train_array,
            dtype=torch.float32,
            device=device,
        )
        X_valid_numeric_tensor = torch.as_tensor(
            X_valid_numeric,
            dtype=torch.float32,
            device=device,
        )
        X_valid_categorical_tensor = torch.as_tensor(
            X_valid_categorical,
            dtype=torch.long,
            device=device,
        )

        n_ens = params["n_ens"]
        train_batch_size = params["train_bs"]
        eval_batch_size = params["eval_bs"]
        epochs = params["epochs"]
        total_steps = epochs * len(y_train_array)
        train_order = np.arange(len(y_train_array))
        best_score = -np.inf
        best_epoch = 0
        best_valid_probabilities = None

        for epoch in range(epochs):
            self.model_.train()
            for start in range(0, len(y_train_array), train_batch_size):
                progress = (epoch * len(y_train_array) + start) / total_steps
                batch_indices = train_order[start : start + train_batch_size]

                for group in optimizer.param_groups:
                    group["lr"] = apply_schedule(
                        group["lr_base"],
                        progress,
                        params["lr_sched"],
                        params["flat_ratio"],
                    )

                optimizer.zero_grad()
                logits = self.model_(
                    X_train_numeric_tensor[batch_indices],
                    X_train_categorical_tensor[batch_indices],
                )
                label_smoothing = apply_schedule(
                    params["ls_eps"],
                    progress,
                    params["ls_eps_sched"],
                    params["flat_ratio"],
                )
                dropout = apply_schedule(
                    params["dropout"],
                    progress,
                    params["p_drop_sched"],
                    params["flat_ratio"],
                )
                for dropout_module in self.model_._dropout_modules:
                    dropout_module.p = dropout

                loss = binary_bce_loss(
                    y_train_tensor[batch_indices].repeat_interleave(n_ens),
                    logits.reshape(-1),
                    label_smoothing=label_smoothing,
                    pos_weight=pos_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(),
                    params["grad_clip"],
                )
                optimizer.step()

            np.random.shuffle(train_order)
            valid_probabilities = self._predict_from_tensors(
                X_valid_numeric_tensor,
                X_valid_categorical_tensor,
                eval_batch_size,
            )
            epoch_score = roc_auc_score(y_valid_array, valid_probabilities)
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_valid_probabilities = valid_probabilities.copy()
                torch.save(self.model_.state_dict(), checkpoint_path)

            if params["verbosity"] >= 2:
                print(
                    f"  epoch {epoch + 1}/{epochs} "
                    f"score={epoch_score:.5f} "
                    f"best={best_score:.5f} "
                    f"ls={label_smoothing:.4f} "
                    f"drop={dropout:.4f}"
                )

        self.model_.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        )
        self.best_score_ = best_score
        self.best_epoch_ = best_epoch
        self.best_valid_probabilities_ = best_valid_probabilities
        self.device_ = device
        return self

    def _predict_from_tensors(
        self,
        numeric_tensor,
        categorical_tensor,
        batch_size,
    ):
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate(
                [
                    torch.sigmoid(
                        self.model_(
                            numeric_tensor[start : start + batch_size],
                            categorical_tensor[start : start + batch_size],
                        )
                    )
                    .mean(dim=1)
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                    for start in range(0, len(numeric_tensor), batch_size)
                ],
                axis=0,
            )

    def predict_proba(self, X):
        X_numeric = self.preprocessor_.transform(
            X[self.numeric_cols_].values.astype(np.float32)
        )
        X_categorical = X[self.categorical_cols_].values.astype(np.int64)
        X_categorical = np.clip(
            X_categorical,
            0,
            np.array(self.categorical_dims_) - 1,
        )
        numeric_tensor = torch.as_tensor(
            X_numeric,
            dtype=torch.float32,
            device=self.device_,
        )
        categorical_tensor = torch.as_tensor(
            X_categorical,
            dtype=torch.long,
            device=self.device_,
        )
        positive_probabilities = self._predict_from_tensors(
            numeric_tensor,
            categorical_tensor,
            self.params["eval_bs"],
        )
        return np.stack(
            [1.0 - positive_probabilities, positive_probabilities],
            axis=1,
        )


def serializable_config():
    result = {}
    for key, value in CONFIG.items():
        result[key] = value.__name__ if isinstance(value, type) else value
    return result


def main():
    seed_everything(SEED)
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was not detected. Enable a Kaggle GPU accelerator before "
            "running the 17th RealMLP experiment."
        )

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, sample, external = load_data()

    train_id = train[ID_COL].copy()
    test_id = test[ID_COL].copy()
    y = train[TARGET_COL].astype(int).reset_index(drop=True)
    y_external = external[TARGET_COL].astype(int).reset_index(drop=True)
    X = train.drop(columns=[ID_COL, TARGET_COL])
    X_test = test.drop(columns=[ID_COL])
    X_external = external.drop(columns=[TARGET_COL])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    feature_engineer = FeatureEngineer(categorical_cols, numeric_cols)
    X, new_categorical_cols, new_numeric_cols, combo_names = (
        feature_engineer.transform(X, fit=True)
    )
    X_test, _, _, _ = feature_engineer.transform(X_test, fit=False)
    X_external, _, _, _ = feature_engineer.transform(X_external, fit=False)
    categorical_cols += new_categorical_cols
    numeric_cols += new_numeric_cols

    gpu_count = torch.cuda.device_count()
    print(f"PyTorch version: {torch.__version__}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {WORK_DIR}")
    print(f"CUDA GPU count: {gpu_count}")
    for gpu_index in range(gpu_count):
        print(f"CUDA GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    print("Using CUDA GPU 0. RealMLP already ensembles 16 models internally.")
    print(f"Competition train shape: {X.shape}")
    print(f"External train shape: {X_external.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Categorical features before fold TE: {len(categorical_cols)}")
    print(f"Numeric features before fold TE: {len(numeric_cols)}")
    print(f"Fold-safe target encoding features: {combo_names}")

    folds = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )
    competition_splits = list(folds.split(X, y))
    external_splits = list(folds.split(X_external, y_external))
    oof_prediction = np.zeros(len(X), dtype=float)
    test_prediction = np.zeros(len(X_test), dtype=float)
    fold_records = []

    for fold, ((train_idx, valid_idx), (external_train_idx, _)) in enumerate(
        zip(competition_splits, external_splits),
        start=1,
    ):
        print("\n" + "=" * 60)
        print(f"RealMLP Fold {fold}/{N_SPLITS}")
        print("=" * 60)

        X_train = pd.concat(
            [
                X.iloc[train_idx].copy(),
                X_external.iloc[external_train_idx].copy(),
            ],
            ignore_index=True,
        )
        y_train = pd.concat(
            [
                y.iloc[train_idx],
                y_external.iloc[external_train_idx],
            ],
            ignore_index=True,
        )
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx]
        X_fold_test = X_test.copy()

        if USE_TARGET_ENCODING:
            target_encoder = TargetEncoder(
                cv=N_SPLITS,
                smooth="auto",
                shuffle=True,
                random_state=SEED,
            )
            train_encoded = target_encoder.fit_transform(
                X_train[combo_names],
                y_train,
            )
            valid_encoded = target_encoder.transform(X_valid[combo_names])
            test_encoded = target_encoder.transform(X_fold_test[combo_names])
            target_encoding_names = [f"_{col}TE" for col in combo_names]
            X_train[target_encoding_names] = train_encoded
            X_valid[target_encoding_names] = valid_encoded
            X_fold_test[target_encoding_names] = test_encoded

        print(f"Features after fold TE: {len(X_train.columns)}")
        model = RealMLPClassifier(**CONFIG)
        model.fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            categorical_cols=categorical_cols,
            checkpoint_path=CHECKPOINT_DIR / f"model_fold_{fold}.pth",
        )
        valid_prediction = model.best_valid_probabilities_
        fold_test_prediction = model.predict_proba(X_fold_test)[:, 1]
        oof_prediction[valid_idx] = valid_prediction
        test_prediction += fold_test_prediction / N_SPLITS

        fold_auc = roc_auc_score(y_valid, valid_prediction)
        fold_records.append(
            {
                "fold": fold,
                "auc": fold_auc,
                "best_epoch": model.best_epoch_,
                "competition_train_rows": len(train_idx),
                "external_train_rows": len(external_train_idx),
                "internal_ensemble_members": CONFIG["n_ens"],
            }
        )
        print(
            f"RealMLP Fold {fold}: AUC={fold_auc:.6f}, "
            f"best_epoch={model.best_epoch_}"
        )

        del (
            model,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_fold_test,
            valid_prediction,
            fold_test_prediction,
        )
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_auc = roc_auc_score(y, oof_prediction)
    fold_metrics = pd.DataFrame(fold_records)
    print("\n" + "=" * 60)
    print("Pre-submission diagnostics")
    print("=" * 60)
    print(f"GPU RealMLP OOF AUC: {ensemble_auc:.6f}")
    print(
        "Fold AUC: "
        f"mean={fold_metrics['auc'].mean():.6f}, "
        f"std={fold_metrics['auc'].std():.6f}, "
        f"min={fold_metrics['auc'].min():.6f}, "
        f"max={fold_metrics['auc'].max():.6f}"
    )
    print("\nTest prediction distribution:")
    print(pd.Series(test_prediction, name=TARGET_COL).describe().to_string())

    pd.DataFrame(
        {
            ID_COL: train_id,
            TARGET_COL: y,
            "oof_prediction": oof_prediction,
        }
    ).to_csv(DIAGNOSTIC_DIR / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(DIAGNOSTIC_DIR / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "gpu_realmlp_oof_auc",
                "competition_train_rows",
                "external_train_rows",
                "internal_ensemble_members",
                "fold_count",
                "gpu_count",
            ],
            "value": [
                ensemble_auc,
                len(X),
                len(X_external),
                CONFIG["n_ens"],
                N_SPLITS,
                gpu_count,
            ],
        }
    ).to_csv(DIAGNOSTIC_DIR / "run_summary.csv", index=False)
    with (DIAGNOSTIC_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(serializable_config(), file, indent=2)

    submission = sample.copy()
    submission[ID_COL] = test_id.values
    submission[TARGET_COL] = np.clip(test_prediction, 0.0, 1.0)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved diagnostics: {DIAGNOSTIC_DIR}")
    print(f"Saved submission: {OUTPUT_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
