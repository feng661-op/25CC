# -*- coding: utf-8 -*-
"""
2025 C 题 问题 3：多因素、检测误差与达标比例联合建模
========================================================

核心原则：
1. 以孕妇为独立统计单位，使用已经审计过的阈值删失表；
2. 用区间删失 Lognormal AFT 比较 BMI 与多因素候选模型，不为题意强行保留无增益变量；
3. 最终仍按 BMI 有序分组；分组目标是让组内个体 95% 预测达标时间尽量接近；
4. 组推荐时点使用组内个体条件 CDF 的混合分布，而不是“代表 BMI 的个人 T95”；
5. 在 10~25 周内取满足目标达标比例 q 的最早时点；主结果 q=95%，并做 90/95/97.5% 敏感性；
6. 测量误差由同次采血重复检测直接估计，并做 Monte Carlo 误差传播；
7. 参数和推荐时点的不确定性按孕妇整簇 bootstrap。

输出目录：问题三/output/
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# --------------------------- 配置 ---------------------------
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "问题三"
OUT = BASE / "output"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MALE_CLEAN = ROOT / "data" / "processed" / "male_cleaned.csv"
CENSOR_FILE = ROOT / "outputs" / "tables" / "male_threshold_censoring.csv"
REPEAT_ERR_FILE = ROOT / "outputs" / "tables" / "male_same_draw_repeat_error_summary.csv"

THRESH = 0.04
T_MIN = 10.0
T_MAX = 25.0
Q_MAIN = 0.95
Q_LEVELS = [0.90, 0.95, 0.975]
MIN_GROUP_N = 35
K_CANDIDATES = [3, 4, 5, 6]
BIC_PARSIMONY_DELTA = 6.0
CV_FOLDS = 5
BOOT_REPS = 150
EDGE_BOOT_REPS = 80
MC_REPS_PER_LEVEL = 60
RANDOM_SEED = 20250901

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180


# --------------------------- 工具函数 ---------------------------
def mode_text(series: pd.Series) -> str:
    m = series.dropna().astype(str).mode()
    return str(m.iloc[0]) if len(m) else ""


def parse_gravidity(v: str) -> int:
    s = str(v).strip()
    if "3" in s:
        return 3
    try:
        return int(float(s))
    except Exception:
        return 1


def risk_grade(t: float | None) -> str:
    if t is None or not np.isfinite(t):
        return "常规窗口内未满足可靠度"
    if t <= 12:
        return "低风险（≤12周）"
    if t <= 27:
        return "高风险（13–27周）"
    return "极高风险（≥28周）"


def safe_round(x, digits=4):
    if x is None or not np.isfinite(x):
        return np.nan
    return round(float(x), digits)


def savefig(fig, filename: str):
    fig.tight_layout()
    fig.savefig(FIG / filename)
    plt.close(fig)


# --------------------------- 1. 数据与孕妇级协变量 ---------------------------
record = pd.read_csv(MALE_CLEAN, encoding="utf-8-sig")
censor = pd.read_csv(CENSOR_FILE, encoding="utf-8-sig")

cov_rows = []
for woman, g in record.groupby("孕妇代码", sort=False):
    g = g.sort_values("孕周_连续值")
    first_ga = float(g["孕周_连续值"].min())
    first = g[np.isclose(g["孕周_连续值"].astype(float), first_ga)]
    cov_rows.append(
        {
            "孕妇代码": woman,
            "age": float(np.nanmedian(pd.to_numeric(g["年龄"], errors="coerce"))),
            "height": float(np.nanmedian(pd.to_numeric(g["身高"], errors="coerce"))),
            "weight": float(np.nanmedian(pd.to_numeric(first["体重"], errors="coerce"))),
            "gravidity": parse_gravidity(mode_text(g["怀孕次数"])),
            "parity": float(np.nanmedian(pd.to_numeric(g["生产次数"], errors="coerce"))),
            "ivf": mode_text(g["IVF妊娠"]),
        }
    )
cov = pd.DataFrame(cov_rows)

subj = (
    censor[["孕妇代码", "baseline_BMI", "删失类型", "threshold_lower", "threshold_upper"]]
    .merge(cov, on="孕妇代码", how="left", validate="one_to_one")
    .rename(
        columns={
            "baseline_BMI": "BMI",
            "删失类型": "ctype",
            "threshold_lower": "lo",
            "threshold_upper": "hi",
        }
    )
    .reset_index(drop=True)
)

if subj["孕妇代码"].duplicated().any():
    raise RuntimeError("孕妇级阈值表存在重复孕妇，停止运行。")
if subj[["BMI", "age", "height", "weight", "gravidity", "parity"]].isna().any().any():
    raise RuntimeError("问题三所需孕妇级协变量存在缺失，停止运行。")

ctype = subj["ctype"].astype(str).to_numpy()
lo_arr = pd.to_numeric(subj["lo"], errors="coerce").to_numpy(float)
hi_arr = pd.to_numeric(subj["hi"], errors="coerce").to_numpy(float)
n_subjects = len(subj)

valid_types = {"left", "interval", "right"}
if set(np.unique(ctype)) - valid_types:
    raise RuntimeError(f"发现未知删失类型：{set(np.unique(ctype)) - valid_types}")


# --------------------------- 2. 共线性审计 ---------------------------
def vif_table(frame: pd.DataFrame, cols: list[str], set_name: str) -> pd.DataFrame:
    x = frame[cols].astype(float).copy()
    x = (x - x.mean()) / x.std(ddof=0)
    vals = []
    for j, col in enumerate(cols):
        vals.append(
            {
                "变量集合": set_name,
                "变量": col,
                "VIF": float(variance_inflation_factor(x.to_numpy(), j)),
            }
        )
    return pd.DataFrame(vals)


vif_full = vif_table(
    subj,
    ["BMI", "age", "height", "weight", "gravidity", "parity"],
    "BMI+年龄+身高+体重+孕次+产次",
)
vif_reduced = vif_table(
    subj,
    ["BMI", "age", "height", "gravidity", "parity"],
    "去除体重后的候选集合",
)
vif_all = pd.concat([vif_full, vif_reduced], ignore_index=True)
vif_all.to_csv(OUT / "02_变量共线性.csv", index=False, encoding="utf-8-sig")

corr_body = subj[["BMI", "height", "weight"]].corr()
corr_body.to_csv(OUT / "02b_BMI身高体重相关矩阵.csv", encoding="utf-8-sig")


# --------------------------- 3. 区间删失 AFT 候选模型 ---------------------------
CONT_VARS = ["BMI", "age", "height", "weight", "parity"]
MEAN = {c: float(subj[c].mean()) for c in CONT_VARS}
STD = {c: float(subj[c].std(ddof=0)) for c in CONT_VARS}


def z(c: str) -> np.ndarray:
    return (subj[c].to_numpy(float) - MEAN[c]) / STD[c]


# 每个候选模型都是真正不同的变量组合；BMI+weight+height 不同时放入，避免代数共线。
MODEL_SPECS = {
    "BMI": ["BMI_z"],
    "BMI+年龄": ["BMI_z", "age_z"],
    "BMI+年龄+身高": ["BMI_z", "age_z", "height_z"],
    "BMI+年龄+产次": ["BMI_z", "age_z", "parity_z"],
    "BMI+年龄+孕次": ["BMI_z", "age_z", "grav2", "grav3p"],
    "BMI+年龄+身高+产次": ["BMI_z", "age_z", "height_z", "parity_z"],
    "BMI+年龄+孕次+辅助生殖": ["BMI_z", "age_z", "grav2", "grav3p", "assisted"],
    "体重+身高+年龄+产次（无BMI）": ["weight_z", "height_z", "age_z", "parity_z"],
}

FEATURES = {
    "BMI_z": z("BMI"),
    "age_z": z("age"),
    "height_z": z("height"),
    "weight_z": z("weight"),
    "parity_z": z("parity"),
    "grav2": (subj["gravidity"].to_numpy(float) == 2).astype(float),
    "grav3p": (subj["gravidity"].to_numpy(float) >= 3).astype(float),
    "assisted": (subj["ivf"].astype(str).to_numpy() != "自然受孕").astype(float),
}


def make_design(features: list[str]) -> np.ndarray:
    arr = [np.ones(n_subjects)] + [FEATURES[f] for f in features]
    return np.column_stack(arr)


def aft_nll(theta: np.ndarray, X: np.ndarray, ct: np.ndarray, lo: np.ndarray, hi: np.ndarray, idx=None) -> float:
    if idx is None:
        idx = np.arange(len(ct))
    idx = np.asarray(idx, dtype=int)
    beta = theta[:-1]
    sigma = float(np.exp(theta[-1]))
    mu = X[idx] @ beta
    c = ct[idx]
    l = lo[idx]
    h = hi[idx]
    lik = np.empty(len(idx), dtype=float)

    mask_l = c == "left"
    mask_r = c == "right"
    mask_i = c == "interval"

    if mask_l.any():
        lik[mask_l] = norm.cdf((np.log(h[mask_l]) - mu[mask_l]) / sigma)
    if mask_r.any():
        lik[mask_r] = norm.sf((np.log(l[mask_r]) - mu[mask_r]) / sigma)
    if mask_i.any():
        zl = (np.log(l[mask_i]) - mu[mask_i]) / sigma
        zh = (np.log(h[mask_i]) - mu[mask_i]) / sigma
        lik[mask_i] = norm.cdf(zh) - norm.cdf(zl)

    return float(-np.sum(np.log(np.clip(lik, 1e-300, None))))


def fit_aft(X: np.ndarray, ct=ctype, lo=lo_arr, hi=hi_arr, idx=None):
    p = X.shape[1]
    init = np.r_[2.4, np.zeros(p - 1), np.log(0.55)]
    res = minimize(
        lambda th: aft_nll(th, X, ct, lo, hi, idx),
        init,
        method="L-BFGS-B",
        bounds=[(None, None)] * p + [(-3.0, 1.0)],
        options={"maxiter": 1200, "ftol": 1e-11},
    )
    return res


def make_cv_folds(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    chunks = np.array_split(order, k)
    folds = []
    all_idx = np.arange(n)
    for test in chunks:
        train = np.setdiff1d(all_idx, test, assume_unique=False)
        folds.append((train, test))
    return folds


cv_folds = make_cv_folds(n_subjects, CV_FOLDS, RANDOM_SEED)
model_rows = []
model_fit = {}
model_X = {}
for model_name, feats in MODEL_SPECS.items():
    X = make_design(feats)
    res = fit_aft(X)
    if not res.success:
        raise RuntimeError(f"候选模型 {model_name} 拟合失败：{res.message}")
    kpar = len(res.x)
    nll = float(res.fun)
    aic = 2.0 * kpar + 2.0 * nll
    bic = math.log(n_subjects) * kpar + 2.0 * nll
    cv_nll = []
    for train, test in cv_folds:
        rr = fit_aft(X, idx=train)
        if not rr.success:
            cv_nll.append(np.nan)
        else:
            cv_nll.append(aft_nll(rr.x, X, ctype, lo_arr, hi_arr, idx=test) / len(test))
    model_rows.append(
        {
            "模型": model_name,
            "特征": "+".join(feats),
            "参数个数_含sigma": kpar,
            "NLL": nll,
            "AIC": aic,
            "BIC": bic,
            "5折孕妇级CV_NLL每人": float(np.nanmean(cv_nll)),
            "sigma_AFT": float(np.exp(res.x[-1])),
            "拟合成功": bool(res.success),
        }
    )
    model_fit[model_name] = res
    model_X[model_name] = X

model_cmp = pd.DataFrame(model_rows).sort_values(["BIC", "AIC"]).reset_index(drop=True)
model_cmp.to_csv(OUT / "01_模型比较.csv", index=False, encoding="utf-8-sig")
selected_model = str(model_cmp.iloc[0]["模型"])
selected_res = model_fit[selected_model]
selected_X = model_X[selected_model]
selected_features = MODEL_SPECS[selected_model]
selected_beta = selected_res.x[:-1]
selected_sigma = float(np.exp(selected_res.x[-1]))
selected_mu = selected_X @ selected_beta

coef_table = pd.DataFrame(
    {
        "参数": ["intercept"] + selected_features + ["sigma_AFT"],
        "估计值": list(selected_beta) + [selected_sigma],
    }
)
coef_table.to_csv(OUT / "01b_主模型参数.csv", index=False, encoding="utf-8-sig")


# --------------------------- 4. BMI 有序分段 ---------------------------
def ordered_segmentation(bmi: np.ndarray, target: np.ndarray, k: int, min_n: int):
    order = np.argsort(bmi, kind="mergesort")
    b = np.asarray(bmi, float)[order]
    y = np.asarray(target, float)[order]
    n = len(b)
    if k * min_n > n:
        return None

    ps = np.r_[0.0, np.cumsum(y)]
    ps2 = np.r_[0.0, np.cumsum(y * y)]

    def seg_cost(i, j):
        nn = j - i
        sy = ps[j] - ps[i]
        sy2 = ps2[j] - ps2[i]
        return max(float(sy2 - sy * sy / nn), 0.0)

    inf = 1e100
    dp = np.full((k + 1, n + 1), inf, dtype=float)
    prev = np.full((k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for kk in range(1, k + 1):
        for j in range(kk * min_n, n + 1):
            lo_i = (kk - 1) * min_n
            hi_i = j - min_n
            if hi_i < lo_i:
                continue
            for i in range(lo_i, hi_i + 1):
                if not np.isfinite(dp[kk - 1, i]):
                    continue
                # 不允许把完全相同的 BMI 从中间切开。
                if i > 0 and i < n and b[i - 1] == b[i]:
                    continue
                val = dp[kk - 1, i] + seg_cost(i, j)
                if val < dp[kk, j]:
                    dp[kk, j] = val
                    prev[kk, j] = i

    if not np.isfinite(dp[k, n]):
        return None

    segs = []
    j = n
    for kk in range(k, 0, -1):
        i = int(prev[kk, j])
        if i < 0:
            return None
        segs.append((i, j))
        j = i
    segs.reverse()

    cuts = []
    for idx in range(len(segs) - 1):
        left_end = segs[idx][1] - 1
        right_start = segs[idx + 1][0]
        cuts.append(float((b[left_end] + b[right_start]) / 2.0))

    sse = float(dp[k, n])
    # 分段常数均值 K 个 + K-1 个切点，使用 BIC 型复杂度惩罚。
    bic_like = n * math.log(max(sse / n, 1e-12)) + (2 * k - 1) * math.log(n)
    return {
        "order": order,
        "bmi_sorted": b,
        "target_sorted": y,
        "segments": segs,
        "cuts": cuts,
        "sse": sse,
        "bic_like": bic_like,
    }


log_tau95 = selected_mu + norm.ppf(Q_MAIN) * selected_sigma
seg_candidates = []
seg_objects = {}
for k in K_CANDIDATES:
    obj = ordered_segmentation(subj["BMI"].to_numpy(float), log_tau95, k, MIN_GROUP_N)
    if obj is None:
        continue
    seg_objects[k] = obj
    seg_candidates.append({"K": k, "SSE_log_tau95": obj["sse"], "BIC_like": obj["bic_like"]})
seg_compare = pd.DataFrame(seg_candidates).sort_values("K")
if seg_compare.empty:
    raise RuntimeError("BMI 有序分组无可行解。")
min_bic_seg = float(seg_compare["BIC_like"].min())
# ΔBIC≤6 时优先更简单的 K，避免在 267 人且高左删失数据上过度切组。
eligible_k = seg_compare.loc[seg_compare["BIC_like"] <= min_bic_seg + BIC_PARSIMONY_DELTA, "K"]
selected_k = int(eligible_k.min())
seg_compare["距最优BIC差"] = seg_compare["BIC_like"] - min_bic_seg
seg_compare["主方案"] = seg_compare["K"].eq(selected_k)
seg_compare.to_csv(OUT / "03a_分组K比较.csv", index=False, encoding="utf-8-sig")
seg_obj = seg_objects[selected_k]
cutpoints = seg_obj["cuts"]


def assign_group(bmi_values: np.ndarray, cuts: list[float]) -> np.ndarray:
    return np.digitize(np.asarray(bmi_values, float), np.asarray(cuts, float), right=False)


group_id = assign_group(subj["BMI"].to_numpy(float), cutpoints)


def mixture_cdf(t: float, mu: np.ndarray, sigma: float) -> float:
    return float(np.mean(norm.cdf((math.log(t) - mu) / sigma)))


def earliest_time(mu: np.ndarray, sigma: float, q: float, tmin=T_MIN, tmax=T_MAX):
    fmin = mixture_cdf(tmin, mu, sigma)
    fmax = mixture_cdf(tmax, mu, sigma)
    if fmin >= q:
        return float(tmin), fmin, fmax, True
    if fmax < q:
        return np.nan, fmin, fmax, False
    root = brentq(lambda tt: mixture_cdf(tt, mu, sigma) - q, tmin, tmax)
    return float(root), fmin, fmax, True


def group_bounds(g: int):
    if g == 0:
        lo = float(subj["BMI"].min())
    else:
        lo = float(cutpoints[g - 1])
    if g == selected_k - 1:
        hi = float(subj["BMI"].max())
    else:
        hi = float(cutpoints[g])
    return lo, hi


group_rows = []
q_rows = []
for g in range(selected_k):
    idx = np.where(group_id == g)[0]
    mu_g = selected_mu[idx]
    lo_bmi, hi_bmi = group_bounds(g)
    t_main, f10, f25, feasible = earliest_time(mu_g, selected_sigma, Q_MAIN)
    group_rows.append(
        {
            "组": g + 1,
            "BMI下界": lo_bmi,
            "BMI上界": hi_bmi,
            "人数": len(idx),
            "组内BMI均值": float(subj.iloc[idx]["BMI"].mean()),
            "10周预测达标比例": f10,
            "12周预测达标比例": mixture_cdf(12.0, mu_g, selected_sigma),
            "25周预测达标比例": f25,
            "95%推荐孕周": t_main,
            "95%目标在10至25周可行": feasible,
            "推荐时点风险等级": risk_grade(t_main),
        }
    )
    for q in Q_LEVELS:
        tq, fq10, fq25, ok = earliest_time(mu_g, selected_sigma, q)
        q_rows.append(
            {
                "组": g + 1,
                "q": q,
                "推荐孕周": tq,
                "10周达标比例": fq10,
                "25周达标比例": fq25,
                "可行": ok,
            }
        )

group_table = pd.DataFrame(group_rows)
q_table = pd.DataFrame(q_rows)
group_table.to_csv(OUT / "03_BMI分组与推荐时点.csv", index=False, encoding="utf-8-sig")
q_table.to_csv(OUT / "04_阈值敏感性.csv", index=False, encoding="utf-8-sig")


# --------------------------- 5. 孕妇级 bootstrap ---------------------------
rng = np.random.default_rng(RANDOM_SEED)
boot_time_rows = []
boot_edge_rows = []
for b_rep in range(BOOT_REPS):
    sample_idx = rng.integers(0, n_subjects, size=n_subjects)
    rr = fit_aft(selected_X, idx=sample_idx)
    if not rr.success:
        continue
    beta_b = rr.x[:-1]
    sigma_b = float(np.exp(rr.x[-1]))
    mu_b_all = selected_X @ beta_b
    sampled_groups = group_id[sample_idx]
    for g in range(selected_k):
        chosen = sample_idx[sampled_groups == g]
        if len(chosen) < 5:
            continue
        t, _, _, ok = earliest_time(mu_b_all[chosen], sigma_b, Q_MAIN)
        boot_time_rows.append(
            {"bootstrap": b_rep, "组": g + 1, "推荐孕周": t, "可行": ok}
        )

# 切点稳定性单独做较少次数的 bootstrap；每次重新按抽中孕妇的 BMI/预测 tau 分段。
for b_rep in range(EDGE_BOOT_REPS):
    sample_idx = rng.integers(0, n_subjects, size=n_subjects)
    rr = fit_aft(selected_X, idx=sample_idx)
    if not rr.success:
        continue
    mu_sample = selected_X[sample_idx] @ rr.x[:-1]
    sigma_sample = float(np.exp(rr.x[-1]))
    target_sample = mu_sample + norm.ppf(Q_MAIN) * sigma_sample
    obj = ordered_segmentation(
        subj["BMI"].to_numpy(float)[sample_idx], target_sample, selected_k, MIN_GROUP_N
    )
    if obj is None or len(obj["cuts"]) != selected_k - 1:
        continue
    for j, cp in enumerate(obj["cuts"]):
        boot_edge_rows.append({"bootstrap": b_rep, "切点序号": j + 1, "BMI切点": cp})

boot_times = pd.DataFrame(boot_time_rows)
boot_summary_rows = []
for g in range(selected_k):
    vals = pd.to_numeric(
        boot_times.loc[(boot_times["组"] == g + 1) & boot_times["可行"], "推荐孕周"], errors="coerce"
    ).dropna()
    total = int((boot_times["组"] == g + 1).sum())
    feasible_n = len(vals)
    boot_summary_rows.append(
        {
            "组": g + 1,
            "bootstrap有效次数": total,
            "可行次数": feasible_n,
            "不可行比例": 1.0 - feasible_n / total if total else np.nan,
            "推荐孕周均值": float(vals.mean()) if feasible_n else np.nan,
            "推荐孕周标准差": float(vals.std(ddof=1)) if feasible_n > 1 else np.nan,
            "推荐孕周2.5%": float(vals.quantile(0.025)) if feasible_n else np.nan,
            "推荐孕周50%": float(vals.quantile(0.5)) if feasible_n else np.nan,
            "推荐孕周97.5%": float(vals.quantile(0.975)) if feasible_n else np.nan,
        }
    )
boot_summary = pd.DataFrame(boot_summary_rows)
boot_summary.to_csv(OUT / "06_bootstrap时点区间.csv", index=False, encoding="utf-8-sig")

edge_boot = pd.DataFrame(boot_edge_rows)
edge_summary_rows = []
if not edge_boot.empty:
    for j in range(1, selected_k):
        vals = edge_boot.loc[edge_boot["切点序号"] == j, "BMI切点"].dropna()
        edge_summary_rows.append(
            {
                "切点序号": j,
                "主样本切点": cutpoints[j - 1],
                "bootstrap次数": len(vals),
                "切点中位数": float(vals.median()) if len(vals) else np.nan,
                "切点2.5%": float(vals.quantile(0.025)) if len(vals) else np.nan,
                "切点97.5%": float(vals.quantile(0.975)) if len(vals) else np.nan,
            }
        )
edge_summary = pd.DataFrame(edge_summary_rows)
edge_summary.to_csv(OUT / "07_切点bootstrap稳定性.csv", index=False, encoding="utf-8-sig")


# --------------------------- 6. 检测误差 Monte Carlo ---------------------------
repeat_err = pd.read_csv(REPEAT_ERR_FILE, encoding="utf-8-sig")
diff_sd = float(repeat_err.iloc[0]["Y差值标准差"])
sigma_y_primary = diff_sd / math.sqrt(2.0)
error_levels = [sigma_y_primary, 0.010, 0.017]

record_by_woman = {
    str(w): g[["孕周_连续值", "Y染色体浓度"]].copy()
    for w, g in record.groupby("孕妇代码", sort=False)
}
woman_order = subj["孕妇代码"].astype(str).tolist()


def censor_from_noisy_y(sigma_y: float, local_rng: np.random.Generator):
    ct = []
    lo = []
    hi = []
    for woman in woman_order:
        g = record_by_woman[woman].copy()
        yy = g["Y染色体浓度"].to_numpy(float) + local_rng.normal(0.0, sigma_y, len(g))
        tmp = pd.DataFrame({"GA": g["孕周_连续值"].to_numpy(float), "Y": yy})
        # 同孕周多次检测先取中位数，避免同一孕周产生零宽度“区间删失”。
        tmp = tmp.groupby("GA", as_index=False)["Y"].median().sort_values("GA")
        ga = tmp["GA"].to_numpy(float)
        y = tmp["Y"].to_numpy(float)
        hits = np.where(y >= THRESH)[0]
        if len(hits) == 0:
            ct.append("right"); lo.append(float(ga[-1])); hi.append(np.nan)
        elif hits[0] == 0:
            ct.append("left"); lo.append(np.nan); hi.append(float(ga[0]))
        else:
            j = int(hits[0])
            ct.append("interval"); lo.append(float(ga[j - 1])); hi.append(float(ga[j]))
    return np.asarray(ct), np.asarray(lo, float), np.asarray(hi, float)


mc_rows = []
for level_i, sigma_y in enumerate(error_levels):
    local_rng = np.random.default_rng(RANDOM_SEED + 1000 + level_i)
    for rep in range(MC_REPS_PER_LEVEL):
        ct_mc, lo_mc, hi_mc = censor_from_noisy_y(sigma_y, local_rng)
        rr = fit_aft(selected_X, ct=ct_mc, lo=lo_mc, hi=hi_mc)
        if not rr.success:
            continue
        mu_mc = selected_X @ rr.x[:-1]
        sigma_aft_mc = float(np.exp(rr.x[-1]))
        for g in range(selected_k):
            idx = np.where(group_id == g)[0]
            t, _, _, ok = earliest_time(mu_mc[idx], sigma_aft_mc, Q_MAIN)
            mc_rows.append(
                {
                    "sigma_Y": sigma_y,
                    "模拟": rep,
                    "组": g + 1,
                    "推荐孕周": t,
                    "可行": ok,
                }
            )

mc = pd.DataFrame(mc_rows)
mc_summary_rows = []
for sigma_y in error_levels:
    for g in range(selected_k):
        block = mc[(np.isclose(mc["sigma_Y"], sigma_y)) & (mc["组"] == g + 1)]
        vals = pd.to_numeric(block.loc[block["可行"], "推荐孕周"], errors="coerce").dropna()
        mc_summary_rows.append(
            {
                "sigma_Y": sigma_y,
                "组": g + 1,
                "模拟有效次数": len(block),
                "不可行比例": 1.0 - len(vals) / len(block) if len(block) else np.nan,
                "推荐孕周均值": float(vals.mean()) if len(vals) else np.nan,
                "推荐孕周标准差": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "推荐孕周2.5%": float(vals.quantile(0.025)) if len(vals) else np.nan,
                "推荐孕周97.5%": float(vals.quantile(0.975)) if len(vals) else np.nan,
            }
        )
mc_summary = pd.DataFrame(mc_summary_rows)
mc_summary.to_csv(OUT / "05_误差敏感性.csv", index=False, encoding="utf-8-sig")


# --------------------------- 7. 图 ---------------------------
fig, ax = plt.subplots(figsize=(8.2, 4.8))
plot_cmp = model_cmp.sort_values("BIC", ascending=True)
ypos = np.arange(len(plot_cmp))
ax.barh(ypos, plot_cmp["BIC"].to_numpy(float))
ax.set_yticks(ypos)
ax.set_yticklabels(plot_cmp["模型"], fontsize=8)
ax.set_xlabel("BIC（越小越好）")
ax.set_title("问题三：多因素 AFT 候选模型比较")
savefig(fig, "fig1_模型BIC比较.png")

fig, ax = plt.subplots(figsize=(8.0, 5.0))
t_grid = np.linspace(T_MIN, T_MAX, 240)
for g in range(selected_k):
    idx = np.where(group_id == g)[0]
    vals = [mixture_cdf(t, selected_mu[idx], selected_sigma) for t in t_grid]
    ax.plot(t_grid, vals, label=f"组{g+1}")
ax.axhline(Q_MAIN, linestyle="--", linewidth=1.2, label="95%可靠度")
ax.set_xlim(T_MIN, T_MAX)
ax.set_ylim(0.45, 1.005)
ax.set_xlabel("孕周（周）")
ax.set_ylabel("模型预测 Y≥4% 的累计达标比例")
ax.set_title("各 BMI 组达标概率曲线（组内混合 CDF）")
ax.legend(ncol=2, fontsize=8)
savefig(fig, "fig2_各组混合CDF.png")

fig, ax = plt.subplots(figsize=(8.0, 4.8))
xx = np.arange(selected_k)
vals = group_table["95%推荐孕周"].to_numpy(float)
ax.bar(xx, np.nan_to_num(vals, nan=T_MAX))
for i, v in enumerate(vals):
    text = f"{v:.1f}周" if np.isfinite(v) else "25周仍不足95%"
    ax.text(i, (v if np.isfinite(v) else T_MAX) + 0.25, text, ha="center", fontsize=9)
ax.axhline(12, linestyle="--", linewidth=1.0, label="12周低/高风险分界")
ax.set_xticks(xx)
ax.set_xticklabels([f"组{i+1}\nn={int(group_table.iloc[i]['人数'])}" for i in range(selected_k)])
ax.set_ylabel("95%可靠度下最早推荐孕周")
ax.set_title("问题三：BMI 分组与推荐 NIPT 时点")
ax.legend(fontsize=8)
savefig(fig, "fig3_分组推荐时点.png")

fig, ax = plt.subplots(figsize=(8.0, 4.8))
for g in range(selected_k):
    block = mc_summary[mc_summary["组"] == g + 1].sort_values("sigma_Y")
    ax.errorbar(
        block["sigma_Y"],
        block["推荐孕周均值"],
        yerr=block["推荐孕周标准差"],
        marker="o",
        capsize=3,
        label=f"组{g+1}",
    )
ax.axvline(sigma_y_primary, linestyle="--", linewidth=1.0, label="重复检测估计σ")
ax.set_xlabel("单次 Y 浓度测量误差 σY")
ax.set_ylabel("95%可靠度推荐孕周（Monte Carlo均值±SD）")
ax.set_title("检测误差对推荐时点的影响")
ax.legend(ncol=2, fontsize=8)
savefig(fig, "fig4_检测误差敏感性.png")

fig, ax = plt.subplots(figsize=(8.0, 4.8))
ax.errorbar(
    boot_summary["组"],
    boot_summary["推荐孕周50%"],
    yerr=np.vstack(
        [
            boot_summary["推荐孕周50%"] - boot_summary["推荐孕周2.5%"],
            boot_summary["推荐孕周97.5%"] - boot_summary["推荐孕周50%"],
        ]
    ),
    fmt="o",
    capsize=4,
)
ax.set_xticks(range(1, selected_k + 1))
ax.set_xlabel("BMI组")
ax.set_ylabel("推荐孕周")
ax.set_title("孕妇级 Bootstrap：推荐时点 95% 区间")
savefig(fig, "fig5_bootstrap推荐时点.png")


# --------------------------- 8. 报告 ---------------------------
def md_table(df: pd.DataFrame, columns: list[str], float_digits=3) -> str:
    out = df[columns].copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep] + rows)


censor_counts = subj["ctype"].value_counts()
full_vif_body = vif_full.set_index("变量")["VIF"].to_dict()
model_best = model_cmp.iloc[0]
model_second = model_cmp.iloc[1]

report_lines = []
report_lines.append("# 2025 C 题 问题三：多因素、检测误差与达标比例联合分析")
report_lines.append("")
report_lines.append("## 1. 题目要求与建模对象")
report_lines.append(
    "题目要求综合考虑身高、体重、年龄等因素、检测误差和 Y 染色体浓度达标比例，并最终仍按 BMI 给出分组与最佳 NIPT 时点。"
)
report_lines.append(
    f"本分析以 {n_subjects} 名男胎孕妇为独立统计单位；当前阈值删失结构为：左删失 {int(censor_counts.get('left',0))} 人、区间删失 {int(censor_counts.get('interval',0))} 人、右删失 {int(censor_counts.get('right',0))} 人。"
)
report_lines.append(
    "真实达到 4% 的时间多数不能精确观察，因此使用区间删失 Lognormal AFT，而不是把首次观测达标孕周直接当真实达标时间。"
)
report_lines.append("")
report_lines.append("## 2. 多因素是否真的需要保留")
report_lines.append(
    "BMI、体重、身高存在结构性关系 BMI=体重/身高²；若三者同时进入线性模型会产生严重多重共线，因此候选模型不把 BMI、体重、身高三者机械同时塞入。"
)
report_lines.append(
    f"全变量 VIF 审计中 BMI={full_vif_body.get('BMI',np.nan):.1f}、体重={full_vif_body.get('weight',np.nan):.1f}、身高={full_vif_body.get('height',np.nan):.1f}，说明该组合不可直接解释系数。"
)
report_lines.append(
    f"候选模型比较后，BIC 最优模型为 **{selected_model}**（BIC={float(model_best['BIC']):.2f}，AIC={float(model_best['AIC']):.2f}）；次优 BIC 为 {model_second['模型']}（BIC={float(model_second['BIC']):.2f}）。"
)
if selected_model == "BMI":
    report_lines.append(
        "因此，本题并不是没有考虑年龄、身高、体重和孕产史，而是这些变量在本数据中未提供足以抵消模型复杂度的稳定增量信息。主模型仍由 BMI 主导；这是数据驱动的模型选择结论，不是事先删除多因素。"
    )
else:
    report_lines.append("多因素候选模型获得了足够的增量拟合证据，因此主模型保留上述变量。")
report_lines.append("")
report_lines.append("### 候选模型比较")
report_lines.append(md_table(model_cmp, ["模型", "NLL", "AIC", "BIC", "5折孕妇级CV_NLL每人"], 3))
report_lines.append("")
report_lines.append("## 3. BMI 有序分组")
report_lines.append(
    f"先由主 AFT 模型得到每名孕妇的 95% 预测达标时间，再按 BMI 顺序做动态规划分段，约束每组至少 {MIN_GROUP_N} 人。比较 K=3~6 后，用 BIC 型准则并在 ΔBIC≤{BIC_PARSIMONY_DELTA:g} 时优先更简单方案，最终选 **K={selected_k}**。"
)
report_lines.append(md_table(seg_compare, ["K", "SSE_log_tau95", "BIC_like", "距最优BIC差", "主方案"], 3))
report_lines.append("")
report_lines.append("## 4. 每组最佳 NIPT 时点")
report_lines.append(
    "对每组使用组内个体条件达标分布的平均 F_G(t)=mean[F_i(t)]。主结果在 10~25 周内寻找满足 F_G(t)≥95% 的最早 t；这同时满足可靠度约束，并因时点尽量早而降低延迟发现风险。"
)
report_lines.append(md_table(group_table, ["组", "BMI下界", "BMI上界", "人数", "12周预测达标比例", "25周预测达标比例", "95%推荐孕周", "推荐时点风险等级"], 3))
report_lines.append("")
report_lines.append("95% 是决策可靠度参数而非题目给定常数，因此同时给 90%/95%/97.5% 敏感性，不把单一阈值伪装成唯一临床真值。")
report_lines.append("")
report_lines.append("## 5. 检测误差")
report_lines.append(
    f"同次采血重复检测的 Y 差值标准差为 {diff_sd:.6f}。若两次重复检测误差近似独立同方差，则单次测量误差 σY≈SD(diff)/√2={sigma_y_primary:.6f}。该值作为主误差场景，同时把 0.010 和原问题二曾使用的 0.017 作为更保守敏感性。"
)
report_lines.append(
    "Monte Carlo 中直接扰动每条 Y 浓度、按同孕周中位数重新构造删失区间、重拟合 AFT，再重新计算各组推荐周，因此误差影响被传播到最终决策，而不是只在终点加误差条。"
)
report_lines.append(md_table(mc_summary, ["sigma_Y", "组", "不可行比例", "推荐孕周均值", "推荐孕周标准差", "推荐孕周2.5%", "推荐孕周97.5%"], 3))
report_lines.append("")
report_lines.append("## 6. 孕妇级 Bootstrap 不确定性")
report_lines.append(md_table(boot_summary, ["组", "不可行比例", "推荐孕周50%", "推荐孕周2.5%", "推荐孕周97.5%"], 3))
if not edge_summary.empty:
    report_lines.append("")
    report_lines.append("### BMI 切点稳定性")
    report_lines.append(md_table(edge_summary, ["切点序号", "主样本切点", "切点中位数", "切点2.5%", "切点97.5%"], 3))
report_lines.append("")
report_lines.append("## 7. 结论边界")
report_lines.append(
    "本数据多数孕妇在首次 NIPT 时已经达到 4%，因此早期达标时间存在大量左删失。AFT 能正确利用这些信息，但 10 周以前的分布主要依赖模型外推；本报告只在题目给出的 10~25 周常规窗口内给出决策。"
)
report_lines.append(
    "若某组在 25 周仍达不到目标可靠度，应报告‘常规窗口内无法满足’，而不是继续外推到 25 周以后。"
)

(OUT / "问题三分析报告.md").write_text("\n".join(report_lines), encoding="utf-8")

summary = {
    "n_subjects": n_subjects,
    "censor_counts": {k: int(v) for k, v in censor_counts.items()},
    "selected_model": selected_model,
    "selected_model_AIC": float(model_best["AIC"]),
    "selected_model_BIC": float(model_best["BIC"]),
    "selected_sigma_AFT": selected_sigma,
    "selected_k": selected_k,
    "cutpoints": [float(x) for x in cutpoints],
    "sigma_y_primary": sigma_y_primary,
    "group_results": group_table.to_dict(orient="records"),
}
(OUT / "运行摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("Q3_DONE")
print("selected_model=", selected_model)
print("selected_K=", selected_k)
print("cutpoints=", [round(x, 4) for x in cutpoints])
print("sigma_Y_primary=", sigma_y_primary)
print(group_table.to_string(index=False))
