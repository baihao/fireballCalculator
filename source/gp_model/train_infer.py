"""K / C 双单任务 GP 训练与推理（GPyTorch），见 gp_fireball_kc_lmc_strategy.md / fireball_gp_mogp_module_design.md。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import gpytorch
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from gp_model.config import merge_hyperparams, save_trained_artifact
from gp_model.data_input import Dataset

logger = logging.getLogger(__name__)

NUM_TASKS = 2  # Dataset Y 列数 K, C；B 为训练集均值 b_mean


def _base_kernel(name: str, d: int) -> gpytorch.kernels.Kernel:
    n = (name or "RBF").upper()
    if n == "RBF":
        return gpytorch.kernels.RBFKernel(ard_num_dims=d)
    if n in ("MATERN", "MATÉRN") or "MATERN" in n:
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
    return gpytorch.kernels.RBFKernel(ard_num_dims=d)


def _gaussian_likelihood_from_cov(
    cov: dict[str, Any],
    device: torch.device,
    *,
    task_label: str,
) -> gpytorch.likelihoods.GaussianLikelihood:
    """
    观测噪声 σ²（Diagonal）：
    - `observation_noise_fixed`：固定方差，不参加优化。
    - `observation_noise_max` / `observation_noise_min`：可学习噪声的区间约束。
    - `observation_noise_init`：可学习时的初值。
    噪声绝对值由 `build_covariance_kc_from_dataset` 结合 `covariance_autoscale`
    与 `observation_noise_*_times_training_{k,c}_var` 自生成。
    """
    fixed = cov.get("observation_noise_fixed")
    if fixed is not None:
        sig2 = float(fixed)
        if sig2 <= 0:
            raise ValueError("observation_noise_fixed 须为正标量方差 σ²")
        lk = gpytorch.likelihoods.GaussianLikelihood().to(device)
        lk.initialize(noise=sig2)
        lk.raw_noise.requires_grad_(False)
        logger.info("[%s] likelihood: 观测噪声固定 σ²=%.6g", task_label, sig2)
        return lk

    nmax = cov.get("observation_noise_max")
    nmin = cov.get("observation_noise_min", 1e-8)
    if nmax is not None:
        lo = float(nmin)
        hi = float(nmax)
        if hi <= lo:
            raise ValueError("observation_noise_max 须大于 observation_noise_min")
        c = gpytorch.constraints.Interval(lo, hi)
        lk = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=c).to(device)
        logger.info("[%s] likelihood: 可学习 σ²∈[%.6g, %.6g]", task_label, lo, hi)
    else:
        lk = gpytorch.likelihoods.GaussianLikelihood().to(device)

    n0 = cov.get("observation_noise_init")
    if n0 is not None:
        lk.initialize(noise=float(n0))
    return lk


def _training_scalar_label_variance_scale(y: torch.Tensor) -> float:
    """ref ≈ 训练标签散布（优先样本方差，退化用 mean(|y|)²）。"""
    yf = y.detach().float().cpu().numpy().ravel()
    v = float(np.var(yf))
    if not np.isfinite(v) or v <= 0:
        m = float(np.mean(np.abs(yf)))
        v = m * m if m > 0 else 1e-18
    return max(v, 1e-18)


def _augment_covariance_lengthscale_floor(
    X_np: np.ndarray,
    cov: dict[str, Any],
) -> dict[str, Any]:
    """与各维跨度对齐：`span_floor = max(配置值, min_d(span_d)*1e-4)`，避免跨度过小维退化。"""
    out = dict(cov)
    span = np.ptp(X_np, axis=0)
    positive = span[span > 1e-15]
    if positive.size == 0:
        data_floor = 1e-6
    else:
        data_floor = max(1e-9, float(np.min(positive)) * 1e-4)
    cfg = float(out.get("length_scale_init_span_floor", 1e-6))
    out["length_scale_init_span_floor"] = max(cfg, data_floor)
    return out


def _resolve_task_covariance_noise(
    cov: dict[str, Any],
    y: torch.Tensor,
    task_label: str,
    autoscale: dict[str, Any],
) -> dict[str, Any]:
    """
    用 **本任务** 训练标签 ref≈var(y) 解析 σ²：
    1) `observation_noise_*_times_training_k_var` / `_c_var`（仅补未手写的绝对键）；
    2) 若仍无 `observation_noise_max` 且未 `fixed`，且 `covariance_autoscale.enabled`（默认 True）：
       σ²_max = max_mult×ref，σ²_init = init_mult×ref，σ²_min = max×1e-6。
    """
    out = dict(cov)
    ref = _training_scalar_label_variance_scale(y)

    if task_label == "K":
        k_max = out.get("observation_noise_max_times_training_k_var")
        k_init = out.get("observation_noise_init_times_training_k_var")
        k_min = out.get("observation_noise_min_times_training_k_var")
    elif task_label == "C":
        k_max = out.get("observation_noise_max_times_training_c_var")
        k_init = out.get("observation_noise_init_times_training_c_var")
        k_min = out.get("observation_noise_min_times_training_c_var")
    else:
        raise ValueError(task_label)

    if k_max is not None and out.get("observation_noise_max") is None:
        out["observation_noise_max"] = float(k_max) * ref
    if k_init is not None and out.get("observation_noise_init") is None:
        out["observation_noise_init"] = float(k_init) * ref
    if k_min is not None and out.get("observation_noise_min") is None:
        out["observation_noise_min"] = float(k_min) * ref

    if out.get("observation_noise_max") is not None and out.get("observation_noise_min") is None:
        out["observation_noise_min"] = max(1e-18, float(out["observation_noise_max"]) * 1e-6)

    if out.get("observation_noise_max") is not None and out.get("observation_noise_min") is not None:
        lo, hi = float(out["observation_noise_min"]), float(out["observation_noise_max"])
        if lo >= hi:
            out["observation_noise_min"] = hi * 1e-3

    if (
        out.get("observation_noise_fixed") is None
        and out.get("observation_noise_max") is None
        and autoscale.get("enabled", True)
    ):
        mm = float(autoscale.get("observation_noise_max_mult", 0.5))
        mi = float(autoscale.get("observation_noise_init_mult", 0.1))
        out["observation_noise_max"] = mm * ref
        if out.get("observation_noise_init") is None:
            out["observation_noise_init"] = mi * ref
        if out.get("observation_noise_min") is None:
            out["observation_noise_min"] = max(1e-18, float(out["observation_noise_max"]) * 1e-6)
        logger.info(
            "[%s] covariance_autoscale: var(y)≈%.6g → σ² max=%.6g init=%.6g min=%.6g",
            task_label,
            ref,
            out["observation_noise_max"],
            out["observation_noise_init"],
            out["observation_noise_min"],
        )

    return out


def build_covariance_kc_from_dataset(
    dataset: Dataset,
    hyperparams: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    根据 **dataset.X**（跨度）、**dataset.Y** 的 K/C 列（标签散布）自主补全
    `covariance_K`、`covariance_C`（合并 `hyperparams` 中已有项，不删用户显式字段）。

    - **length_scale_init_span_floor**：`max(原配置, min_d(span_d)×1e-4)`；
    - **观测噪声**：见 `covariance_autoscale` 与 `observation_noise_*_times_training_{k,c}_var`。
    """
    X_np = np.asarray(dataset.X, dtype=np.float64)
    Y_np = np.asarray(dataset.Y, dtype=np.float64)
    if Y_np.shape[1] != NUM_TASKS:
        raise ValueError(f"Y 须为 (n,{NUM_TASKS}) [K,C]，当前 {Y_np.shape}")

    cov_k = dict(hyperparams.get("covariance_K") or {})
    cov_c = dict(hyperparams.get("covariance_C") or {})
    cov_k = _augment_covariance_lengthscale_floor(X_np, cov_k)
    cov_c = _augment_covariance_lengthscale_floor(X_np, cov_c)

    autoscale = hyperparams.get("covariance_autoscale") or {}
    yk = torch.tensor(Y_np[:, 0], dtype=torch.float32)
    yc = torch.tensor(Y_np[:, 1], dtype=torch.float32)
    cov_k = _resolve_task_covariance_noise(cov_k, yk, "K", autoscale)
    cov_c = _resolve_task_covariance_noise(cov_c, yc, "C", autoscale)
    return cov_k, cov_c


class SingleTaskGPModel(gpytorch.models.ExactGP):
    """单输出：输入 x∈R^d，标量 y。"""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        *,
        cov: dict[str, Any],
    ):
        super().__init__(train_x, train_y, likelihood)
        d = int(train_x.shape[-1])
        self.mean_module = gpytorch.means.ConstantMean()
        base = _base_kernel(str(cov.get("input_kernel", "RBF")), d)
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


TaskName = str


def _init_single_task_lengthscale(
    model: SingleTaskGPModel,
    X: torch.Tensor,
    cov: dict[str, Any],
    *,
    task_label: TaskName,
) -> None:
    """ScaleKernel(base=RBF)；lengthscale 在 base_kernel 上 initialize。"""
    d = int(X.shape[-1])
    device, dtype = X.device, X.dtype
    base_kern = model.covar_module.base_kernel
    mode = str(cov.get("length_scale_init_mode", "span_fraction")).lower()
    glob = float(cov.get("length_scale_init", 1.0))

    if mode == "fixed":
        val = max(glob, 1e-12)
        target = torch.full((1, d), val, dtype=dtype, device=device)
        base_kern.initialize(lengthscale=target)
        logger.info("[%s] lengthscale 初值(fixed): 各维=%.6g", task_label, val)
        return

    if mode != "span_fraction":
        raise ValueError(
            f"未知 length_scale_init_mode={mode!r}，请用 span_fraction 或 fixed"
        )

    span = X.max(dim=0).values - X.min(dim=0).values
    floor = float(cov.get("length_scale_init_span_floor", 1e-6))
    span = torch.clamp(span, min=floor)
    alpha = float(cov.get("length_scale_init_span_fraction", 0.2))
    target = (glob * alpha * span).view(1, -1).to(dtype=dtype, device=device)
    base_kern.initialize(lengthscale=target)
    span_np = span.detach().cpu().numpy()
    ell_np = target.detach().cpu().numpy().ravel()
    logger.info(
        "[%s] lengthscale 初值(span_fraction α=%s glob=%s): span=%s → ℓ=%s",
        task_label,
        alpha,
        glob,
        np.array2string(span_np, precision=6),
        np.array2string(ell_np, precision=6),
    )


def _stack_kbc_mean(mean_kc: np.ndarray, b_mean: float) -> np.ndarray:
    m = mean_kc.shape[0]
    mean = np.zeros((m, 3), dtype=np.float64)
    mean[:, 0] = mean_kc[:, 0]
    mean[:, 1] = b_mean
    mean[:, 2] = mean_kc[:, 1]
    return mean


def _stack_kbc_variance(var_kc: np.ndarray) -> np.ndarray:
    m = var_kc.shape[0]
    var = np.zeros((m, 3), dtype=np.float64)
    var[:, 0] = var_kc[:, 0]
    var[:, 2] = var_kc[:, 1]
    return var


@dataclass
class TrainedDualGP:
    model_K: SingleTaskGPModel
    likelihood_K: gpytorch.likelihoods.GaussianLikelihood
    model_C: SingleTaskGPModel
    likelihood_C: gpytorch.likelihoods.GaussianLikelihood
    hyperparams: dict[str, Any]
    train_x: torch.Tensor
    train_y: torch.Tensor
    b_mean: float


# 兼容旧名称
TrainedMOGP = TrainedDualGP


def _optimization_restart_seeds(opt_cfg: dict[str, Any]) -> list[int]:
    explicit = opt_cfg.get("restart_seeds")
    if explicit is not None:
        seeds = [int(s) for s in explicit]
        if not seeds:
            raise ValueError("restart_seeds 为非空整数列表")
        return seeds
    n = int(opt_cfg.get("num_restarts", 1))
    if n < 1:
        n = 1
    base = int(opt_cfg.get("restart_base_seed", 0))
    return [base + k for k in range(n)]


def _marginal_log_likelihood_scalar(
    mll: ExactMarginalLogLikelihood,
    model: SingleTaskGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    model.train()
    likelihood.train()
    with torch.no_grad():
        out = model(X)
        return float(mll(out, y).item())


def _train_one_scalar_task(
    X: torch.Tensor,
    y: torch.Tensor,
    cov: dict[str, Any],
    opt_cfg: dict[str, Any],
    dev: torch.device,
    task_label: TaskName,
) -> tuple[
    SingleTaskGPModel,
    gpytorch.likelihoods.GaussianLikelihood,
    float,
    int,
    list[int],
]:
    lr = float(opt_cfg.get("learning_rate", 0.05))
    n_iter = int(opt_cfg.get("max_iter", 200))
    seeds = _optimization_restart_seeds(opt_cfg)

    best_mll = float("-inf")
    best_model_sd: dict[str, torch.Tensor] | None = None
    best_likelihood_sd: dict[str, torch.Tensor] | None = None
    best_restart = 0

    for r, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        likelihood = _gaussian_likelihood_from_cov(cov, dev, task_label=task_label)
        model = SingleTaskGPModel(X, y, likelihood, cov=cov).to(dev)
        _init_single_task_lengthscale(model, X, cov, task_label=task_label)

        model.train()
        likelihood.train()
        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(n_iter):
            optimizer.zero_grad()
            out = model(X)
            loss = -mll(out, y)
            loss.backward()
            optimizer.step()
            if len(seeds) == 1:
                if (i + 1) % max(1, n_iter // 10) == 0 or i == 0:
                    logger.info(
                        "[%s] train iter %s/%s loss=%.4f",
                        task_label,
                        i + 1,
                        n_iter,
                        float(loss.item()),
                    )
            else:
                if i == 0 or (i + 1) == n_iter:
                    logger.info(
                        "[%s] restart %s/%s seed=%s iter %s/%s loss=%.4f",
                        task_label,
                        r + 1,
                        len(seeds),
                        seed,
                        i + 1,
                        n_iter,
                        float(loss.item()),
                    )

        score = _marginal_log_likelihood_scalar(mll, model, likelihood, X, y)
        logger.info(
            "[%s] restart %s/%s seed=%s marginal_log_likelihood=%.6f",
            task_label,
            r + 1,
            len(seeds),
            seed,
            score,
        )
        if score > best_mll:
            best_mll = score
            best_restart = r
            best_model_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_likelihood_sd = {
                k: v.detach().cpu().clone() for k, v in likelihood.state_dict().items()
            }

    assert best_model_sd is not None and best_likelihood_sd is not None

    likelihood = _gaussian_likelihood_from_cov(cov, dev, task_label=task_label)
    model.load_state_dict({k: v.to(dev) for k, v in best_model_sd.items()})
    likelihood.load_state_dict({k: v.to(dev) for k, v in best_likelihood_sd.items()})
    model.eval()
    likelihood.eval()

    if len(seeds) > 1:
        logger.info(
            "[%s] 多重启：选 restart_index=%s seed=%s MLL=%.6f",
            task_label,
            best_restart,
            seeds[best_restart],
            best_mll,
        )

    return model, likelihood, best_mll, best_restart, seeds


def train_dual_gp(
    dataset: Dataset,
    hyperparams: dict[str, Any],
    *,
    device: torch.device | None = None,
) -> TrainedDualGP:
    """在原始物理单位上分别训练 K-GP 与 C-GP；不对 X、Y 标准化。"""
    dev = device or torch.device("cpu")
    X_np = np.asarray(dataset.X, dtype=np.float64)
    Y_np = np.asarray(dataset.Y, dtype=np.float64)
    if Y_np.shape[1] != NUM_TASKS:
        raise ValueError(f"Y 须为 (n,{NUM_TASKS})，列为 [K, C]，当前 {Y_np.shape}")

    X = torch.tensor(X_np, dtype=torch.float32, device=dev)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=dev)
    y_k = Y[:, 0].contiguous()
    y_c = Y[:, 1].contiguous()
    n = X.shape[0]
    if n < 2:
        logger.warning("训练样本数 < 2，GP 可能不稳定")

    logger.info(
        "训练：双单任务 GP；X 原单位 (当量 kg, 含铝 %%)，Y=[K,C]；b_mean=%.8g；"
        "covariance_K/C 已由数据自生成（见 build_covariance_kc_from_dataset）",
        dataset.b_mean,
    )

    cov_k, cov_c = build_covariance_kc_from_dataset(dataset, hyperparams)
    opt_cfg = hyperparams["optimization"]

    model_k, lik_k, mll_k, br_k, seeds_k = _train_one_scalar_task(
        X, y_k, cov_k, opt_cfg, dev, "K"
    )
    model_c, lik_c, mll_c, br_c, seeds_c = _train_one_scalar_task(
        X, y_c, cov_c, opt_cfg, dev, "C"
    )

    hp_out = merge_hyperparams(hyperparams)
    hp_out.pop("covariance", None)
    hp_out["covariance_K"] = dict(cov_k)
    hp_out["covariance_C"] = dict(cov_c)
    hp_out["covariance_K"]["optimized"] = True
    hp_out["covariance_C"]["optimized"] = True
    opt = dict(hp_out.get("optimization", {}))
    for legacy in ("best_restart_index", "best_marginal_log_likelihood", "restart_seeds_used"):
        opt.pop(legacy, None)
    opt["K"] = {
        "best_marginal_log_likelihood": float(mll_k),
        "best_restart_index": int(br_k),
        "restart_seeds_used": [int(s) for s in seeds_k],
    }
    opt["C"] = {
        "best_marginal_log_likelihood": float(mll_c),
        "best_restart_index": int(br_c),
        "restart_seeds_used": [int(s) for s in seeds_c],
    }
    hp_out["optimization"] = opt
    hp_out["schema_version"] = hp_out.get("schema_version", 3)
    hp_out["model_family"] = "dual_single_output_gp"

    return TrainedDualGP(
        model_K=model_k,
        likelihood_K=lik_k,
        model_C=model_c,
        likelihood_C=lik_c,
        hyperparams=hp_out,
        train_x=X.detach(),
        train_y=Y.detach(),
        b_mean=float(dataset.b_mean),
    )


train_mogp = train_dual_gp


@dataclass
class PredictiveResult:
    mean: np.ndarray
    variance: np.ndarray
    std: np.ndarray
    variance_latent: np.ndarray
    std_latent: np.ndarray


def predict_mogp(
    trained: TrainedDualGP | dict[str, Any],
    X_star: np.ndarray,
    *,
    device: torch.device | None = None,
) -> PredictiveResult:
    """X_star (m,2) 原单位；返回 K,B,C（B 为常数 b_mean）；K、C 为独立边缘预测。"""
    dev = device or torch.device("cpu")
    if isinstance(trained, dict):
        trained = trained_bundle_from_artifact(trained, device=dev)

    Xs = np.asarray(X_star, dtype=np.float64)
    xs = torch.tensor(Xs, dtype=torch.float32, device=dev)

    trained.model_K.eval()
    trained.likelihood_K.eval()
    trained.model_C.eval()
    trained.likelihood_C.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        lat_k = trained.model_K(xs)
        pred_k = trained.likelihood_K(lat_k)
        lat_c = trained.model_C(xs)
        pred_c = trained.likelihood_C(lat_c)

    mean_k = pred_k.mean.cpu().numpy().reshape(-1, 1)
    mean_c = pred_c.mean.cpu().numpy().reshape(-1, 1)
    mean_kc = np.concatenate([mean_k, mean_c], axis=1)

    var_y_k = pred_k.variance.cpu().numpy().reshape(-1, 1)
    var_y_c = pred_c.variance.cpu().numpy().reshape(-1, 1)
    var_y_kc = np.concatenate([var_y_k, var_y_c], axis=1)

    var_f_k = lat_k.variance.cpu().numpy().reshape(-1, 1)
    var_f_c = lat_c.variance.cpu().numpy().reshape(-1, 1)
    var_f_kc = np.concatenate([var_f_k, var_f_c], axis=1)

    mean = _stack_kbc_mean(mean_kc, trained.b_mean)
    var_y = _stack_kbc_variance(var_y_kc)
    var_f = _stack_kbc_variance(var_f_kc)

    std_y = np.sqrt(np.maximum(var_y, 0.0))
    std_f = np.sqrt(np.maximum(var_f, 0.0))
    return PredictiveResult(
        mean=mean,
        variance=var_y,
        std=std_y,
        variance_latent=var_f,
        std_latent=std_f,
    )


def trained_bundle_from_artifact(
    artifact: dict[str, Any], *, device: torch.device
) -> TrainedDualGP:
    tr = artifact.get("training")
    if not tr or "X" not in tr or "Y" not in tr:
        raise ValueError("训练产物缺少 training.X / training.Y")

    if artifact.get("input_standardization") or artifact.get("output_standardization"):
        raise ValueError(
            "检测到旧版训练产物（含 input/output_standardization）。请重新 train。"
        )
    if "b_mean" not in artifact:
        raise ValueError("训练产物缺少 b_mean，请重新 train。")

    Y_raw = np.asarray(tr["Y"], dtype=np.float64)
    if Y_raw.ndim != 2 or Y_raw.shape[1] != NUM_TASKS:
        raise ValueError(
            f"training.Y 须为 (n,{NUM_TASKS}) [K,C]；若为旧版 3 列，请重新 train。"
        )

    X_raw = np.asarray(tr["X"], dtype=np.float64)
    X = torch.tensor(X_raw, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_raw, dtype=torch.float32, device=device)
    y_k = Y[:, 0].contiguous()
    y_c = Y[:, 1].contiguous()

    cov_k = artifact["covariance_K"]
    cov_c = artifact["covariance_C"]

    torch_blob = artifact["torch"]
    lik_k = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model_k = SingleTaskGPModel(X, y_k, lik_k, cov=cov_k).to(device)
    model_k.load_state_dict(torch_blob["model_K_state_dict"])
    lik_k.load_state_dict(torch_blob["likelihood_K_state_dict"])

    lik_c = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model_c = SingleTaskGPModel(X, y_c, lik_c, cov=cov_c).to(device)
    model_c.load_state_dict(torch_blob["model_C_state_dict"])
    lik_c.load_state_dict(torch_blob["likelihood_C_state_dict"])

    model_k.eval()
    lik_k.eval()
    model_c.eval()
    lik_c.eval()

    return TrainedDualGP(
        model_K=model_k,
        likelihood_K=lik_k,
        model_C=model_c,
        likelihood_C=lik_c,
        hyperparams=artifact,
        train_x=X,
        train_y=Y,
        b_mean=float(artifact["b_mean"]),
    )


def run_train_cli(
    dataset: Dataset,
    hyperparams: dict[str, Any],
    *,
    out_path: str,
    data_dir: str,
) -> None:
    trained = train_dual_gp(dataset, hyperparams)
    X_list = dataset.X.tolist()
    Y_list = dataset.Y.tolist()
    save_trained_artifact(
        out_path,
        trained.hyperparams,
        training_X=X_list,
        training_Y=Y_list,
        data_dir=data_dir,
        b_mean=trained.b_mean,
        model_K_state=trained.model_K.state_dict(),
        likelihood_K_state=trained.likelihood_K.state_dict(),
        model_C_state=trained.model_C.state_dict(),
        likelihood_C_state=trained.likelihood_C.state_dict(),
    )
    logger.info("已写入训练产物: %s", out_path)
