# -*- coding: utf-8 -*-
r"""
2025 C 题 问题 4：女胎 T13/T18/T21 异常综合判定
=================================================

严格按本题第四问流程实现：
1. 监督标签使用附件 AB 列（清洗表中的 abnormal_T13/T18/T21），不是 AE 健康结局；
2. 每条检测记录做预测，但训练/验证始终按“孕妇代码”整体分组，避免重复检测泄漏；
3. M0 = 单 |Z_k| 阈值基准；M1 = Z13/Z18/Z21/ZX/X浓度 Ridge；
   M2 = M1 + GC + 测序质量 + 孕周 + BMI 的 16 特征 Ridge；
4. 外层 5-fold 孕妇级多标签平衡分组得到严格 OOF 预测；Ridge 在外层训练集内部
   再做 3-fold 孕妇级 CV 选择 C(=1/lambda)，并用 inner OOF 概率选择 F1 阈值；
5. 主优化指标为 PR-AUC，分类阈值指标报告 Sensitivity/Precision/F1/Specificity，
   辅助报告 ROC-AUC 和 Brier；
6. 95% CI 使用孕妇整簇 cluster bootstrap，不把 605 条记录当成独立样本；
7. Ridge 只有“总体 PR-AUC 超过 M0 且至少 4/5 个外层 fold 的 PR-AUC 更高”才允许成为最终模型；
8. CV 只用于性能报告；方案确定后再用全部 605 条记录重新拟合最终判定器。

运行：
    D:\mypython\python.exe 问题四\code\q4_pipeline.py

输出：问题四/output/
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- 配置 ---------------------------
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "问题四"
OUT = BASE / "output"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DATA_FILE = ROOT / "data" / "processed" / "female_cleaned.csv"
RANDOM_SEED = 20250903
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOT_REPS = 1000
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
MODEL_PASS_FOLD_WINS = 4

TARGETS = {
    "T13": "abnormal_T13",
    "T18": "abnormal_T18",
    "T21": "abnormal_T21",
}
TARGET_Z = {
    "T13": "Z13",
    "T18": "Z18",
    "T21": "Z21",
}
GROUP_COL = "孕妇代码"
AB_COL = "染色体的非整倍体"

FEATURE_SOURCE = {
    "Z13": "13号染色体的Z值",
    "Z18": "18号染色体的Z值",
    "Z21": "21号染色体的Z值",
    "ZX": "X染色体的Z值",
    "CX": "X染色体浓度",
    "GC": "GC含量",
    "GC13": "13号染色体的GC含量",
    "GC18": "18号染色体的GC含量",
    "GC21": "21号染色体的GC含量",
    "Reads_log": "原始读段数",
    "Map": "在参考基因组上比对的比例",
    "Duplicate": "重复读段的比例",
    "Unique_log": "唯一比对的读段数",
    "Filter": "被过滤掉读段数的比例",
    "GW": "孕周_连续值",
    "BMI": "measurement_BMI",
}

FEATURE_LABEL = {
    "Z13": "Z13",
    "Z18": "Z18",
    "Z21": "Z21",
    "ZX": "ZX",
    "CX": "X染色体浓度",
    "GC": "总体GC",
    "GC13": "GC13",
    "GC18": "GC18",
    "GC21": "GC21",
    "Reads_log": "log(1+原始读段数)",
    "Map": "比对比例",
    "Duplicate": "重复读段比例",
    "Unique_log": "log(1+唯一比对读段数)",
    "Filter": "过滤读段比例",
    "GW": "检测孕周",
    "BMI": "检测时BMI",
}

M1_FEATURES = ["Z13", "Z18", "Z21", "ZX", "CX"]
M2_FEATURES = [
    "Z13", "Z18", "Z21", "ZX", "CX",
    "GC", "GC13", "GC18", "GC21",
    "Reads_log", "Map", "Duplicate", "Unique_log", "Filter", "GW", "BMI",
]
MODEL_FEATURES = {"M1": M1_FEATURES, "M2": M2_FEATURES}
MODEL_NAMES = {
    "M0": "M0 单|Z|阈值基准",
    "M1": "M1 染色体信息Ridge",
    "M2": "M2 16特征综合Ridge",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 180


# --------------------------- 通用工具 ---------------------------
def savefig(fig, filename: str):
    fig.tight_layout()
    fig.savefig(FIG / filename, bbox_inches="tight")
    plt.close(fig)


def safe_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def md_table(df: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    x = df[columns].copy()
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]):
            x[c] = x[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.{digits}f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in x.astype(str).to_numpy()]
    return "\n".join([header, sep] + rows)


def classification_metrics(y, rank_score, pred, probability):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    rank_score = np.asarray(rank_score, float)
    probability = np.asarray(probability, float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if np.isfinite(precision) and np.isfinite(sensitivity) and (precision + sensitivity) else 0.0
    pr_auc = average_precision_score(y, rank_score) if y.sum() > 0 else np.nan
    roc_auc = roc_auc_score(y, rank_score) if len(np.unique(y)) == 2 else np.nan
    brier = brier_score_loss(y, np.clip(probability, 0.0, 1.0)) if np.isfinite(probability).all() else np.nan
    return {
        "n": int(len(y)),
        "positive": int(y.sum()),
        "Sensitivity": safe_float(sensitivity),
        "Precision": safe_float(precision),
        "Specificity": safe_float(specificity),
        "F1": safe_float(f1),
        "PR_AUC": safe_float(pr_auc),
        "ROC_AUC": safe_float(roc_auc),
        "Brier": safe_float(brier),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def choose_f1_threshold(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if len(score) == 0:
        return np.nan, np.nan
    candidates = np.unique(score)
    candidates = np.r_[0.0, candidates]
    best = None
    for t in candidates:
        pred = (score >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        pre = precision_score(y, pred, zero_division=0)
        # F1 主目标；平手优先召回，再优先 precision，最后取更小阈值。
        key = (f1, rec, pre, -float(t))
        if best is None or key > best[0]:
            best = (key, float(t), float(f1))
    return best[1], best[2]


def choose_fbeta_threshold(y, score, beta: float = 2.0):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if len(score) == 0:
        return np.nan, np.nan
    candidates = np.r_[0.0, np.unique(score)]
    best = None
    for t in candidates:
        pred = (score >= t).astype(int)
        fb = fbeta_score(y, pred, beta=beta, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        pre = precision_score(y, pred, zero_division=0)
        key = (fb, rec, pre, -float(t))
        if best is None or key > best[0]:
            best = (key, float(t), float(fb))
    return best[1], best[2]


def build_ridge(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l2", C=float(C), solver="lbfgs", max_iter=5000,
            class_weight=None,
        )),
    ])


def build_baseline_calibrator() -> Pipeline:
    # 仅用于给单 |Z| 基准提供概率/Brier；分类决策仍完全由训练折内优化的 |Z| 阈值决定。
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l2", C=1000.0, solver="lbfgs", max_iter=3000)),
    ])


# --------------------------- 1. 数据载入与特征 ---------------------------
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig").reset_index(drop=True)
if len(df) != 605 or df[GROUP_COL].nunique() != 147:
    raise RuntimeError(f"女胎数据结构与已审计事实不一致：records={len(df)}, women={df[GROUP_COL].nunique()}")

for task, col in TARGETS.items():
    if col not in df.columns:
        raise RuntimeError(f"缺失监督标签列 {col}")

X = pd.DataFrame(index=df.index)
for name, source in FEATURE_SOURCE.items():
    if source not in df.columns:
        raise RuntimeError(f"缺失特征源列：{source}")
    vals = pd.to_numeric(df[source], errors="coerce")
    if name in {"Reads_log", "Unique_log"}:
        vals = np.log1p(vals.clip(lower=0))
    X[name] = vals.astype(float)

Y = pd.DataFrame({task: pd.to_numeric(df[col], errors="raise").astype(int) for task, col in TARGETS.items()})
if not set(np.unique(Y.to_numpy())) <= {0, 1}:
    raise RuntimeError("AB 拆分标签不是二元 0/1。")

# 数据质量审计：不因极端值删样本，只检查缺失及有限性。
missing_counts = X.isna().sum()
if (missing_counts > 0).any():
    print("提示：存在缺失值，将严格在训练折内以中位数填补：")
    print(missing_counts[missing_counts > 0].to_string())

# --------------------------- 2. 孕妇级多标签平衡分折 ---------------------------
def make_group_folds(data: pd.DataFrame, n_splits: int, seed: int, trials: int = 240):
    meta = data.groupby(GROUP_COL, sort=True).agg(
        records=(GROUP_COL, "size"),
        T13=(TARGETS["T13"], "max"),
        T18=(TARGETS["T18"], "max"),
        T21=(TARGETS["T21"], "max"),
    ).reset_index()
    labels = meta[["T13", "T18", "T21"]].to_numpy(float)
    recs = meta["records"].to_numpy(float)
    n = len(meta)
    if n_splits > n:
        raise ValueError("n_splits 大于孕妇数")
    label_totals = labels.sum(axis=0)
    if np.any(label_totals < n_splits):
        raise RuntimeError(f"无法保证每折都有阳性孕妇：totals={label_totals}, folds={n_splits}")
    target_label = label_totals / n_splits
    target_n = n / n_splits
    target_rec = recs.sum() / n_splits
    max_n = int(math.ceil(target_n))
    rarity = 1.0 / np.maximum(label_totals, 1.0)
    priority_base = labels @ rarity + labels.sum(axis=1) * 0.01 + recs / max(recs.max(), 1.0) * 0.001

    def loss(fl, fn, fr):
        label_loss = np.sum((fl - target_label) ** 2 / np.maximum(target_label, 0.5))
        n_loss = 0.25 * np.sum((fn - target_n) ** 2 / max(target_n, 1.0))
        r_loss = 0.08 * np.sum((fr - target_rec) ** 2 / max(target_rec, 1.0))
        return float(label_loss + n_loss + r_loss)

    best = None
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial * 9973)
        jitter = rng.normal(0.0, 0.015, n)
        order = np.argsort(-(priority_base + jitter), kind="mergesort")
        fl = np.zeros((n_splits, 3), float)
        fn = np.zeros(n_splits, float)
        fr = np.zeros(n_splits, float)
        assign = np.full(n, -1, int)
        for i in order:
            costs = []
            for f in range(n_splits):
                if fn[f] >= max_n:
                    costs.append(np.inf)
                    continue
                fl2 = fl.copy(); fn2 = fn.copy(); fr2 = fr.copy()
                fl2[f] += labels[i]
                fn2[f] += 1
                fr2[f] += recs[i]
                costs.append(loss(fl2, fn2, fr2) + rng.uniform(0, 1e-7))
            f = int(np.argmin(costs))
            if not np.isfinite(costs[f]):
                raise RuntimeError("多标签分组器无可行 fold")
            assign[i] = f
            fl[f] += labels[i]
            fn[f] += 1
            fr[f] += recs[i]
        zero_penalty = 1e5 * float(np.sum(fl == 0))
        final_loss = loss(fl, fn, fr) + zero_penalty
        if best is None or final_loss < best[0]:
            best = (final_loss, assign.copy(), fl.copy(), fn.copy(), fr.copy())
    _, assign, fl, fn, fr = best
    mapping = dict(zip(meta[GROUP_COL].astype(str), assign.astype(int)))
    return mapping, meta, fl, fn, fr


outer_map, outer_meta, outer_woman_labels, outer_woman_n, outer_record_n = make_group_folds(
    df, OUTER_FOLDS, RANDOM_SEED
)
outer_fold = df[GROUP_COL].astype(str).map(outer_map).to_numpy(int)
if (outer_fold < 0).any():
    raise RuntimeError("外层 fold 映射失败。")

fold_rows = []
for f in range(OUTER_FOLDS):
    idx = np.where(outer_fold == f)[0]
    women = df.iloc[idx][GROUP_COL].nunique()
    row = {"fold": f + 1, "records": len(idx), "women": women}
    for task in TARGETS:
        row[f"{task}_positive_records"] = int(Y.iloc[idx][task].sum())
        row[f"{task}_positive_women"] = int(df.iloc[idx].assign(_y=Y.iloc[idx][task].to_numpy()).groupby(GROUP_COL)["_y"].max().sum())
    fold_rows.append(row)
fold_distribution = pd.DataFrame(fold_rows)
if (fold_distribution[[f"{t}_positive_records" for t in TARGETS]] == 0).any().any():
    raise RuntimeError(f"外层某 fold 缺少阳性记录，分层失败：\n{fold_distribution}")


# --------------------------- 3. 严格外层 OOF ---------------------------
N = len(df)
oof = {}
for model in ["M0", "M1", "M2"]:
    for task in TARGETS:
        oof[(model, task)] = {
            "rank": np.full(N, np.nan),
            "prob": np.full(N, np.nan),
            "pred": np.full(N, -1, int),
            "threshold": np.full(N, np.nan),
            "pred_f2": np.full(N, -1, int),
            "threshold_f2": np.full(N, np.nan),
            "C": np.full(N, np.nan),
        }

hyper_rows = []

for f in range(OUTER_FOLDS):
    test_idx = np.where(outer_fold == f)[0]
    train_idx = np.where(outer_fold != f)[0]
    train_df = df.iloc[train_idx].copy()
    # inner 分折只在外层训练孕妇中建立，并同时平衡三个标签。
    inner_map, *_ = make_group_folds(train_df, INNER_FOLDS, RANDOM_SEED + 100 + f * 1000)
    inner_fold = train_df[GROUP_COL].astype(str).map(inner_map).to_numpy(int)

    for task in TARGETS:
        y_all = Y[task].to_numpy(int)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        # M0：单 |Z_k| 阈值基准。阈值只在外层 train 选。
        zname = TARGET_Z[task]
        train_score = np.abs(X.iloc[train_idx][zname].to_numpy(float))
        test_score = np.abs(X.iloc[test_idx][zname].to_numpy(float))
        t0, train_f1 = choose_f1_threshold(y_train, train_score)
        t0_f2, train_f2 = choose_fbeta_threshold(y_train, train_score, beta=2.0)
        pcal = build_baseline_calibrator()
        pcal.fit(train_score.reshape(-1, 1), y_train)
        test_prob = pcal.predict_proba(test_score.reshape(-1, 1))[:, 1]
        test_pred = (test_score >= t0).astype(int)
        test_pred_f2 = (test_score >= t0_f2).astype(int)
        block = oof[("M0", task)]
        block["rank"][test_idx] = test_score
        block["prob"][test_idx] = test_prob
        block["pred"][test_idx] = test_pred
        block["threshold"][test_idx] = t0
        block["pred_f2"][test_idx] = test_pred_f2
        block["threshold_f2"][test_idx] = t0_f2
        hyper_rows.append({
            "outer_fold": f + 1, "task": task, "model": "M0", "C": np.nan,
            "lambda": np.nan, "inner_PR_AUC": np.nan, "threshold": t0,
            "inner_or_train_F1_at_threshold": train_f1,
            "threshold_F2": t0_f2, "inner_or_train_F2_at_threshold": train_f2,
        })

        # M1 / M2：在 outer train 内用 inner OOF 选择 C 和阈值。
        for model in ["M1", "M2"]:
            feats = MODEL_FEATURES[model]
            c_results = []
            c_oof = {}
            for C in C_GRID:
                inner_prob = np.full(len(train_idx), np.nan)
                for inf in range(INNER_FOLDS):
                    inner_valid_pos = np.where(inner_fold == inf)[0]
                    inner_train_pos = np.where(inner_fold != inf)[0]
                    pipe = build_ridge(C)
                    pipe.fit(X.iloc[train_idx[inner_train_pos]][feats], y_train[inner_train_pos])
                    inner_prob[inner_valid_pos] = pipe.predict_proba(X.iloc[train_idx[inner_valid_pos]][feats])[:, 1]
                if np.isnan(inner_prob).any():
                    raise RuntimeError(f"inner OOF 概率未填满：fold={f}, task={task}, model={model}, C={C}")
                ap = average_precision_score(y_train, inner_prob)
                c_results.append((float(ap), float(C)))
                c_oof[float(C)] = inner_prob
            # PR-AUC 最大；平手优先更强正则（较小 C）。
            c_results.sort(key=lambda x: (-x[0], x[1]))
            best_ap, best_C = c_results[0]
            best_inner_prob = c_oof[best_C]
            threshold, inner_f1 = choose_f1_threshold(y_train, best_inner_prob)
            threshold_f2, inner_f2 = choose_fbeta_threshold(y_train, best_inner_prob, beta=2.0)
            final_outer = build_ridge(best_C)
            final_outer.fit(X.iloc[train_idx][feats], y_train)
            p_test = final_outer.predict_proba(X.iloc[test_idx][feats])[:, 1]
            pred_test = (p_test >= threshold).astype(int)
            pred_test_f2 = (p_test >= threshold_f2).astype(int)
            block = oof[(model, task)]
            block["rank"][test_idx] = p_test
            block["prob"][test_idx] = p_test
            block["pred"][test_idx] = pred_test
            block["threshold"][test_idx] = threshold
            block["pred_f2"][test_idx] = pred_test_f2
            block["threshold_f2"][test_idx] = threshold_f2
            block["C"][test_idx] = best_C
            hyper_rows.append({
                "outer_fold": f + 1, "task": task, "model": model, "C": best_C,
                "lambda": 1.0 / best_C, "inner_PR_AUC": best_ap,
                "threshold": threshold, "inner_or_train_F1_at_threshold": inner_f1,
                "threshold_F2": threshold_f2, "inner_or_train_F2_at_threshold": inner_f2,
            })

# 完整性检查
for key, block in oof.items():
    if (
        np.isnan(block["rank"]).any()
        or np.isnan(block["prob"]).any()
        or (block["pred"] < 0).any()
        or (block["pred_f2"] < 0).any()
    ):
        raise RuntimeError(f"OOF 预测不完整：{key}")

hyper_table = pd.DataFrame(hyper_rows)


# --------------------------- 4. OOF 评价、fold 稳定性与模型选择 ---------------------------
perf_rows = []
fold_perf_rows = []
for task in TARGETS:
    y = Y[task].to_numpy(int)
    for model in ["M0", "M1", "M2"]:
        block = oof[(model, task)]
        met = classification_metrics(y, block["rank"], block["pred"], block["prob"])
        perf_rows.append({"task": task, "model": model, "model_name": MODEL_NAMES[model], **met})
        for f in range(OUTER_FOLDS):
            idx = np.where(outer_fold == f)[0]
            fm = classification_metrics(y[idx], block["rank"][idx], block["pred"][idx], block["prob"][idx])
            fold_perf_rows.append({"task": task, "model": model, "fold": f + 1, **fm})

perf = pd.DataFrame(perf_rows)
fold_perf = pd.DataFrame(fold_perf_rows)

# F2 阈值敏感性：模型和排序分数不变，仅在各训练层内部改用 F2 选择阈值，观察漏检/误报权衡。
f2_rows = []
for task in TARGETS:
    y = Y[task].to_numpy(int)
    for model in ["M0", "M1", "M2"]:
        block = oof[(model, task)]
        for policy, pred_key in [("F1阈值（主分析）", "pred"), ("F2阈值（召回敏感性）", "pred_f2")]:
            pred = block[pred_key]
            met = classification_metrics(y, block["rank"], pred, block["prob"])
            f2_rows.append({
                "task": task,
                "model": model,
                "threshold_policy": policy,
                "Sensitivity": met["Sensitivity"],
                "Precision": met["Precision"],
                "Specificity": met["Specificity"],
                "F1": met["F1"],
                "F2": float(fbeta_score(y, pred, beta=2.0, zero_division=0)),
            })
f2_sensitivity = pd.DataFrame(f2_rows)

selection_rows = []
selected_model = {}
for task in TARGETS:
    base_pr = float(perf[(perf.task == task) & (perf.model == "M0")]["PR_AUC"].iloc[0])
    candidates = []
    for model in ["M1", "M2"]:
        row = perf[(perf.task == task) & (perf.model == model)].iloc[0]
        model_pr = float(row["PR_AUC"])
        wins = 0
        for f in range(1, OUTER_FOLDS + 1):
            p0 = float(fold_perf[(fold_perf.task == task) & (fold_perf.model == "M0") & (fold_perf.fold == f)]["PR_AUC"].iloc[0])
            pm = float(fold_perf[(fold_perf.task == task) & (fold_perf.model == model) & (fold_perf.fold == f)]["PR_AUC"].iloc[0])
            if np.isfinite(pm) and np.isfinite(p0) and pm > p0:
                wins += 1
        passed = bool(model_pr > base_pr and wins >= MODEL_PASS_FOLD_WINS)
        selection_rows.append({
            "task": task, "candidate": model, "PR_AUC": model_pr,
            "M0_PR_AUC": base_pr, "delta_PR_AUC_vs_M0": model_pr - base_pr,
            "outer_fold_PR_AUC_wins_vs_M0": wins,
            "required_wins": MODEL_PASS_FOLD_WINS,
            "pass_predeclared_rule": passed,
        })
        if passed:
            candidates.append((model_pr, model))
    if candidates:
        candidates.sort(reverse=True)
        selected_model[task] = candidates[0][1]
    else:
        selected_model[task] = "M0"

selection = pd.DataFrame(selection_rows)

# 敏感性分析 1：X 染色体浓度是否带来增益。
# 对 M1 去掉 CX 后重新走与主分析相同的 outer/inner 孕妇级嵌套 CV，而不是在完整数据上直接拟合比较。
def nested_ridge_oof_for_features(feats: list[str]):
    result = {task: np.full(N, np.nan) for task in TARGETS}
    for f in range(OUTER_FOLDS):
        test_idx = np.where(outer_fold == f)[0]
        train_idx = np.where(outer_fold != f)[0]
        train_df = df.iloc[train_idx].copy()
        inner_map, *_ = make_group_folds(train_df, INNER_FOLDS, RANDOM_SEED + 100 + f * 1000)
        inner_fold = train_df[GROUP_COL].astype(str).map(inner_map).to_numpy(int)
        for task in TARGETS:
            y_all = Y[task].to_numpy(int)
            y_train = y_all[train_idx]
            candidates = []
            for C in C_GRID:
                inner_prob = np.full(len(train_idx), np.nan)
                for inf in range(INNER_FOLDS):
                    va_pos = np.where(inner_fold == inf)[0]
                    tr_pos = np.where(inner_fold != inf)[0]
                    pipe = build_ridge(C)
                    pipe.fit(X.iloc[train_idx[tr_pos]][feats], y_train[tr_pos])
                    inner_prob[va_pos] = pipe.predict_proba(X.iloc[train_idx[va_pos]][feats])[:, 1]
                candidates.append((float(average_precision_score(y_train, inner_prob)), float(C)))
            candidates.sort(key=lambda z: (-z[0], z[1]))
            best_C = candidates[0][1]
            pipe = build_ridge(best_C)
            pipe.fit(X.iloc[train_idx][feats], y_train)
            result[task][test_idx] = pipe.predict_proba(X.iloc[test_idx][feats])[:, 1]
    return result

m1_no_cx_oof = nested_ridge_oof_for_features(["Z13", "Z18", "Z21", "ZX"])
x_sens_rows = []
quality_sens_rows = []
for task in TARGETS:
    y = Y[task].to_numpy(int)
    pr_no_cx = float(average_precision_score(y, m1_no_cx_oof[task]))
    pr_with_cx = float(perf[(perf.task == task) & (perf.model == "M1")]["PR_AUC"].iloc[0])
    pr_full = float(perf[(perf.task == task) & (perf.model == "M2")]["PR_AUC"].iloc[0])
    x_sens_rows.append({
        "task": task,
        "M1_without_CX_PR_AUC": pr_no_cx,
        "M1_with_CX_PR_AUC": pr_with_cx,
        "delta_PR_AUC_after_adding_CX": pr_with_cx - pr_no_cx,
    })
    quality_sens_rows.append({
        "task": task,
        "M1_chromosome_PR_AUC": pr_with_cx,
        "M2_full_PR_AUC": pr_full,
        "delta_PR_AUC_after_GC_quality_GW_BMI": pr_full - pr_with_cx,
    })
x_sensitivity = pd.DataFrame(x_sens_rows)
quality_sensitivity = pd.DataFrame(quality_sens_rows)

# 把 fold 稳定性信息并回主结果表。
perf["fold_PR_AUC_wins_vs_M0"] = np.nan
perf["selected_final"] = False
for task in TARGETS:
    for model in ["M1", "M2"]:
        wins = int(selection[(selection.task == task) & (selection.candidate == model)]["outer_fold_PR_AUC_wins_vs_M0"].iloc[0])
        perf.loc[(perf.task == task) & (perf.model == model), "fold_PR_AUC_wins_vs_M0"] = wins
    perf.loc[(perf.task == task) & (perf.model == selected_model[task]), "selected_final"] = True


# --------------------------- 5. 孕妇级 cluster bootstrap 95% CI ---------------------------
group_indices = {str(g): np.asarray(idx, int) for g, idx in df.groupby(GROUP_COL).groups.items()}
women = np.array(sorted(group_indices.keys()), dtype=object)
rng = np.random.default_rng(RANDOM_SEED + 8888)
boot_store = {(task, model): [] for task in TARGETS for model in ["M0", "M1", "M2"]}
metric_names = ["Sensitivity", "Precision", "Specificity", "F1", "PR_AUC", "ROC_AUC", "Brier"]

for b in range(BOOT_REPS):
    sampled = rng.choice(women, size=len(women), replace=True)
    idx = np.concatenate([group_indices[str(w)] for w in sampled])
    for task in TARGETS:
        y = Y[task].to_numpy(int)[idx]
        for model in ["M0", "M1", "M2"]:
            block = oof[(model, task)]
            met = classification_metrics(y, block["rank"][idx], block["pred"][idx], block["prob"][idx])
            boot_store[(task, model)].append([met[m] for m in metric_names])

ci_rows = []
for task in TARGETS:
    for model in ["M0", "M1", "M2"]:
        point = perf[(perf.task == task) & (perf.model == model)].iloc[0]
        arr = np.asarray(boot_store[(task, model)], float)
        row = {"task": task, "model": model, "model_name": MODEL_NAMES[model]}
        for j, m in enumerate(metric_names):
            vals = arr[:, j]
            vals = vals[np.isfinite(vals)]
            row[m] = safe_float(point[m])
            row[f"{m}_CI_low"] = safe_float(np.quantile(vals, 0.025)) if len(vals) else np.nan
            row[f"{m}_CI_high"] = safe_float(np.quantile(vals, 0.975)) if len(vals) else np.nan
        ci_rows.append(row)
ci_table = pd.DataFrame(ci_rows)
selected_ci = pd.concat([
    ci_table[(ci_table.task == task) & (ci_table.model == selected_model[task])]
    for task in TARGETS
], ignore_index=True)


# --------------------------- 6. 全数据重拟合最终模型 ---------------------------
def tune_full_ridge(task: str, model: str):
    y = Y[task].to_numpy(int)
    feats = MODEL_FEATURES[model]
    full_map, *_ = make_group_folds(df, OUTER_FOLDS, RANDOM_SEED + 5000 + (1 if model == "M1" else 2) * 100 + list(TARGETS).index(task))
    full_fold = df[GROUP_COL].astype(str).map(full_map).to_numpy(int)
    results = []
    probs_by_c = {}
    for C in C_GRID:
        p = np.full(N, np.nan)
        for f in range(OUTER_FOLDS):
            va = np.where(full_fold == f)[0]
            tr = np.where(full_fold != f)[0]
            pipe = build_ridge(C)
            pipe.fit(X.iloc[tr][feats], y[tr])
            p[va] = pipe.predict_proba(X.iloc[va][feats])[:, 1]
        ap = average_precision_score(y, p)
        results.append((float(ap), float(C)))
        probs_by_c[float(C)] = p
    results.sort(key=lambda z: (-z[0], z[1]))
    best_ap, best_C = results[0]
    oof_prob = probs_by_c[best_C]
    threshold, f1 = choose_f1_threshold(y, oof_prob)
    pipe = build_ridge(best_C)
    pipe.fit(X[feats], y)
    return {
        "C": best_C,
        "lambda": 1.0 / best_C,
        "cv_PR_AUC": best_ap,
        "threshold": threshold,
        "cv_F1_at_threshold": f1,
        "pipeline": pipe,
        "oof_prob_for_tuning": oof_prob,
    }

final_ridge = {}
for task in TARGETS:
    for model in ["M1", "M2"]:
        final_ridge[(task, model)] = tune_full_ridge(task, model)

final_baseline = {}
for task in TARGETS:
    y = Y[task].to_numpy(int)
    score = np.abs(X[TARGET_Z[task]].to_numpy(float))
    threshold, f1 = choose_f1_threshold(y, score)
    cal = build_baseline_calibrator()
    cal.fit(score.reshape(-1, 1), y)
    final_baseline[task] = {
        "threshold": threshold,
        "train_F1_at_threshold": f1,
        "calibrator": cal,
        "score": score,
        "prob": cal.predict_proba(score.reshape(-1, 1))[:, 1],
    }

# 全量拟合后的操作型最终输出；绝不拿它替代 OOF 性能。
final_pred = pd.DataFrame({
    "序号": df["序号"].to_numpy(),
    GROUP_COL: df[GROUP_COL].astype(str).to_numpy(),
    "AB原始判定": df[AB_COL].fillna("").astype(str).to_numpy(),
    "GC含量": pd.to_numeric(df["GC含量"], errors="coerce").to_numpy(float),
})
final_binary = {}
for task in TARGETS:
    model = selected_model[task]
    y = Y[task].to_numpy(int)
    if model == "M0":
        info = final_baseline[task]
        score = info["score"]
        prob = info["prob"]
        threshold = float(info["threshold"])
        pred = (score >= threshold).astype(int)
        final_pred[f"{task}_score"] = score
        final_pred[f"{task}_probability"] = prob
        final_pred[f"{task}_threshold"] = threshold
    else:
        info = final_ridge[(task, model)]
        prob = info["pipeline"].predict_proba(X[MODEL_FEATURES[model]])[:, 1]
        threshold = float(info["threshold"])
        pred = (prob >= threshold).astype(int)
        final_pred[f"{task}_score"] = prob
        final_pred[f"{task}_probability"] = prob
        final_pred[f"{task}_threshold"] = threshold
    final_pred[f"{task}_selected_model"] = model
    final_pred[f"{task}_true_AB_label"] = y
    final_pred[f"{task}_pred"] = pred
    final_binary[task] = pred

pred_ab = []
for i in range(N):
    tags = [task for task in ["T13", "T18", "T21"] if int(final_binary[task][i]) == 1]
    pred_ab.append("".join(tags) if tags else "无异常")
final_pred["最终预测AB"] = pred_ab
final_pred["检测质量等级"] = np.where(
    final_pred["GC含量"].between(0.40, 0.60, inclusive="both"),
    "GC范围内",
    "GC范围外，可信度降低",
)

# 最终 Ridge 系数：均为标准化后的每 1 SD 系数与 OR，只解释统计关联，不作因果解释。
coef_rows = []
final_param_rows = []
for task in TARGETS:
    b = final_baseline[task]
    final_param_rows.append({
        "task": task, "model": "M0", "selected_final": selected_model[task] == "M0",
        "C": np.nan, "lambda": np.nan, "threshold": b["threshold"],
        "full_groupCV_PR_AUC_for_tuning": np.nan,
    })
    for model in ["M1", "M2"]:
        info = final_ridge[(task, model)]
        pipe = info["pipeline"]
        beta = pipe.named_steps["model"].coef_[0]
        intercept = float(pipe.named_steps["model"].intercept_[0])
        final_param_rows.append({
            "task": task, "model": model, "selected_final": selected_model[task] == model,
            "C": info["C"], "lambda": info["lambda"], "threshold": info["threshold"],
            "full_groupCV_PR_AUC_for_tuning": info["cv_PR_AUC"],
        })
        coef_rows.append({
            "task": task, "model": model, "feature": "intercept", "feature_label": "截距",
            "standardized_beta": intercept, "OR_per_1SD": np.nan,
        })
        for feat, bb in zip(MODEL_FEATURES[model], beta):
            coef_rows.append({
                "task": task, "model": model, "feature": feat, "feature_label": FEATURE_LABEL[feat],
                "standardized_beta": float(bb), "OR_per_1SD": float(np.exp(bb)),
            })
coef_table = pd.DataFrame(coef_rows)
final_params = pd.DataFrame(final_param_rows)


# --------------------------- 7. 数据结构、特征与单变量探索表 ---------------------------
ab_text = df[AB_COL].fillna("").astype(str)
combo_count = int(sum(sum(tag in s for tag in TARGETS) >= 2 for s in ab_text))
change_women = int(df.assign(_ab=ab_text.replace("", "正常")).groupby(GROUP_COL)["_ab"].nunique().gt(1).sum())
structure_rows = [
    {"指标": "女胎检测记录数", "数值": N},
    {"指标": "女胎孕妇数", "数值": df[GROUP_COL].nunique()},
    {"指标": "T13阳性记录数", "数值": int(Y["T13"].sum())},
    {"指标": "T18阳性记录数", "数值": int(Y["T18"].sum())},
    {"指标": "T21阳性记录数", "数值": int(Y["T21"].sum())},
    {"指标": "T13阳性孕妇数", "数值": int(df.assign(_y=Y["T13"]).groupby(GROUP_COL)["_y"].max().sum())},
    {"指标": "T18阳性孕妇数", "数值": int(df.assign(_y=Y["T18"]).groupby(GROUP_COL)["_y"].max().sum())},
    {"指标": "T21阳性孕妇数", "数值": int(df.assign(_y=Y["T21"]).groupby(GROUP_COL)["_y"].max().sum())},
    {"指标": "复合异常记录数", "数值": combo_count},
    {"指标": "AB标签随检测发生变化的孕妇数", "数值": change_women},
    {"指标": "GC处于0.40~0.60记录数", "数值": int(pd.to_numeric(df["GC含量"], errors="coerce").between(0.4, 0.6).sum())},
]
structure = pd.DataFrame(structure_rows)

feature_rows = []
for feat in M2_FEATURES:
    category = (
        "Z值" if feat in {"Z13", "Z18", "Z21", "ZX"} else
        "性染色体" if feat == "CX" else
        "GC" if feat in {"GC", "GC13", "GC18", "GC21"} else
        "测序质量" if feat in {"Reads_log", "Map", "Duplicate", "Unique_log", "Filter"} else
        "孕妇因素"
    )
    transform = "log(1+x)" if feat in {"Reads_log", "Unique_log"} else "原值"
    feature_rows.append({
        "类别": category, "模型变量": feat, "论文名称": FEATURE_LABEL[feat],
        "源字段": FEATURE_SOURCE[feat], "变换": transform,
        "缺失数": int(X[feat].isna().sum()),
    })
feature_table = pd.DataFrame(feature_rows)

univ_rows = []
for task in TARGETS:
    y = Y[task].to_numpy(int)
    for feat in M2_FEATURES:
        v = X[feat].to_numpy(float)
        # 单变量探索只作描述：方向由全样本 AUC 决定，不能作为后续 CV 外的特征筛选。
        mask = np.isfinite(v)
        auc_raw = roc_auc_score(y[mask], v[mask])
        if auc_raw >= 0.5:
            oriented = v[mask]
            direction = "+"
            auc = auc_raw
        else:
            oriented = -v[mask]
            direction = "-"
            auc = 1.0 - auc_raw
        ap = average_precision_score(y[mask], oriented)
        univ_rows.append({
            "task": task, "feature": feat, "feature_label": FEATURE_LABEL[feat],
            "direction_for_exploration": direction, "ROC_AUC_oriented": auc,
            "PR_AUC_oriented": ap, "positive_rate": float(y.mean()),
        })
univ = pd.DataFrame(univ_rows)


# --------------------------- 8. OOF / 输出表 ---------------------------
oof_table = pd.DataFrame({
    "序号": df["序号"].to_numpy(),
    GROUP_COL: df[GROUP_COL].astype(str).to_numpy(),
    "outer_fold": outer_fold + 1,
    "AB原始判定": ab_text.to_numpy(),
})
for task in TARGETS:
    oof_table[f"{task}_true"] = Y[task].to_numpy(int)
    for model in ["M0", "M1", "M2"]:
        block = oof[(model, task)]
        oof_table[f"{task}_{model}_rank"] = block["rank"]
        oof_table[f"{task}_{model}_prob"] = block["prob"]
        oof_table[f"{task}_{model}_pred"] = block["pred"]
        oof_table[f"{task}_{model}_threshold"] = block["threshold"]
        oof_table[f"{task}_{model}_pred_F2"] = block["pred_f2"]
        oof_table[f"{task}_{model}_threshold_F2"] = block["threshold_f2"]

oof_table.to_csv(OUT / "09_OOF预测.csv", index=False, encoding="utf-8-sig")
structure.to_csv(OUT / "01_女胎数据结构.csv", index=False, encoding="utf-8-sig")
feature_table.to_csv(OUT / "02_特征定义.csv", index=False, encoding="utf-8-sig")
univ.to_csv(OUT / "03_单变量探索结果.csv", index=False, encoding="utf-8-sig")
perf.to_csv(OUT / "04_三层模型OOF结果.csv", index=False, encoding="utf-8-sig")
selected_ci.to_csv(OUT / "05_最终性能及置信区间.csv", index=False, encoding="utf-8-sig")
ci_table.to_csv(OUT / "06_全模型cluster_bootstrap_CI.csv", index=False, encoding="utf-8-sig")
fold_distribution.to_csv(OUT / "07_外层fold分布.csv", index=False, encoding="utf-8-sig")
fold_perf.to_csv(OUT / "07b_外层fold性能.csv", index=False, encoding="utf-8-sig")
hyper_table.to_csv(OUT / "08_外层超参数.csv", index=False, encoding="utf-8-sig")
selection.to_csv(OUT / "10_最终模型选择.csv", index=False, encoding="utf-8-sig")
coef_table.to_csv(OUT / "11_最终Ridge系数.csv", index=False, encoding="utf-8-sig")
final_pred.to_csv(OUT / "12_全数据最终判定.csv", index=False, encoding="utf-8-sig")
final_params.to_csv(OUT / "13_最终模型参数.csv", index=False, encoding="utf-8-sig")
f2_sensitivity.to_csv(OUT / "14_F2阈值敏感性.csv", index=False, encoding="utf-8-sig")
x_sensitivity.to_csv(OUT / "15_X染色体浓度敏感性.csv", index=False, encoding="utf-8-sig")
quality_sensitivity.to_csv(OUT / "16_测序质量变量敏感性.csv", index=False, encoding="utf-8-sig")


# --------------------------- 9. 图 ---------------------------
# 图1：T13/T18/T21 阳性/阴性对应 |Z| 分布。
fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
for ax, task in zip(axes, TARGETS):
    y = Y[task].to_numpy(int)
    z = np.abs(X[TARGET_Z[task]].to_numpy(float))
    ax.boxplot([z[y == 0], z[y == 1]], showfliers=True)
    ax.set_xticks([1, 2], ["阴性", "阳性"])
    ax.set_title(f"{task}: |{TARGET_Z[task]}| 分布")
    ax.set_ylabel("|Z|" if task == "T13" else "")
fig.suptitle("女胎 AB 标签与对应染色体 |Z| 分布", y=1.02)
savefig(fig, "fig1_Z值阳性阴性分布.png")

# 图2：PR 曲线，M0/M1/M2 正面对比。
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
for ax, task in zip(axes, TARGETS):
    y = Y[task].to_numpy(int)
    for model in ["M0", "M1", "M2"]:
        score = oof[(model, task)]["rank"]
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(recall, precision, label=f"{model} AP={ap:.3f}")
    ax.axhline(y.mean(), linestyle="--", linewidth=1.0, label=f"随机基线={y.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision" if task == "T13" else "")
    ax.set_title(task)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
fig.suptitle("严格 OOF Precision–Recall 曲线")
savefig(fig, "fig2_PR曲线_M0_M1_M2.png")

# 图3：最终所选模型的 OOF 概率校准。
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
for ax, task in zip(axes, TARGETS):
    y = Y[task].to_numpy(int)
    model = selected_model[task]
    p = oof[(model, task)]["prob"]
    frac, meanp = calibration_curve(y, p, n_bins=5, strategy="quantile")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="理想校准")
    ax.plot(meanp, frac, marker="o", label=f"{task}-{model}")
    ax.set_xlabel("平均预测概率")
    ax.set_ylabel("实际阳性比例" if task == "T13" else "")
    ax.set_title(f"{task}（{model}）")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
fig.suptitle("最终判定模型的 OOF 概率校准（T21 仅作描述）")
savefig(fig, "fig3_最终模型校准曲线.png")

# 图4：每个任务表现更好的 Ridge（M1/M2）的标准化系数。
fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.0))
for ax, task in zip(axes, TARGETS):
    p1 = float(perf[(perf.task == task) & (perf.model == "M1")]["PR_AUC"].iloc[0])
    p2 = float(perf[(perf.task == task) & (perf.model == "M2")]["PR_AUC"].iloc[0])
    model = "M2" if p2 >= p1 else "M1"
    block = coef_table[(coef_table.task == task) & (coef_table.model == model) & (coef_table.feature != "intercept")].copy()
    block["abs"] = block["standardized_beta"].abs()
    block = block.sort_values("abs", ascending=True).tail(12)
    ax.barh(block["feature_label"], block["standardized_beta"])
    ax.axvline(0, linewidth=0.8)
    ax.set_title(f"{task} 最佳Ridge={model}")
    ax.set_xlabel("标准化系数")
fig.suptitle("Ridge 系数方向（统计关联，不代表因果）")
savefig(fig, "fig4_Ridge标准化系数.png")


# --------------------------- 10. 自动报告 ---------------------------
report_perf = perf[["task", "model", "Sensitivity", "Precision", "F1", "PR_AUC", "ROC_AUC", "Brier", "fold_PR_AUC_wins_vs_M0", "selected_final"]].copy()
select_display = pd.DataFrame([
    {
        "task": task,
        "最终模型": selected_model[task],
        "OOF_PR_AUC": float(perf[(perf.task == task) & (perf.model == selected_model[task])]["PR_AUC"].iloc[0]),
        "OOF_F1": float(perf[(perf.task == task) & (perf.model == selected_model[task])]["F1"].iloc[0]),
        "说明": "Ridge满足预设稳定增益规则" if selected_model[task] != "M0" else "Ridge未满足稳定超越M0规则，保留单Z基准",
    }
    for task in TARGETS
])

ci_display = selected_ci[[
    "task", "model", "PR_AUC", "PR_AUC_CI_low", "PR_AUC_CI_high",
    "Sensitivity", "Sensitivity_CI_low", "Sensitivity_CI_high",
    "Precision", "Precision_CI_low", "Precision_CI_high",
    "F1", "F1_CI_low", "F1_CI_high",
]].copy()
selected_f2_display = pd.concat([
    f2_sensitivity[(f2_sensitivity.task == task) & (f2_sensitivity.model == selected_model[task])]
    for task in TARGETS
], ignore_index=True)

report = f"""# 问题四分析报告：女胎 T13/T18/T21 异常综合判定

## 1. 任务定义

本问题预测的是**附件 AB 列的当次 NIPT 非整倍体判读结果**，不是用 AE 列出生后健康状态作为临床真值。AB 被拆为三个并行二元标签：T13、T18、T21。每一条检测记录是一条预测，但同一孕妇的全部记录必须整体进入同一训练折或验证折。

## 2. 数据与标签

{md_table(structure, ["指标", "数值"], digits=0)}

女胎共有 605 条检测记录、147 位孕妇。重复检测保留，因为第四问判定对象是“当次检测”，且有 {change_women} 位孕妇的 AB 标签随检测发生变化。

## 3. 模型体系

- **M0：单 Z 基准。** 对目标染色体使用 $|Z_k|$，每个外层训练折内部选择使 F1 最大的阈值；外层测试折完全不可见。
- **M1：染色体信息 Ridge。** 使用 Z13、Z18、Z21、ZX、X 染色体浓度。
- **M2：16 特征综合 Ridge。** 在 M1 基础上加入总体/染色体 GC、reads、比对/重复/过滤比例、孕周和检测时 BMI；原始读段数与唯一读段数使用 $\\log(1+x)$。
- Ridge 的中位数填补和标准化均在训练折拟合；验证折只应用训练折参数。
- 外层为 5-fold 孕妇级多标签平衡分组；每个外层训练折内部再做 3-fold 孕妇级 CV，以 PR-AUC 选择 $C=1/\\lambda$，再用 inner OOF 概率选择 F1 阈值。
- Accuracy 不作为主指标。主排序指标是 PR-AUC；同时报告 Sensitivity、Precision、F1、Specificity、ROC-AUC、Brier。

### 外层 fold 分布

{md_table(fold_distribution, list(fold_distribution.columns), digits=0)}

## 4. 严格 OOF 结果

{md_table(report_perf, list(report_perf.columns), digits=3)}

随机分类器的 PR-AUC 基准约等于各任务阳性率，因此 T13/T18/T21 的 PR-AUC 应分别结合自己的 prevalence 解释，而不能拿 0.5 作为统一基线。

## 5. 预先规定的模型保留规则

Ridge 只有同时满足以下两点才进入最终判定：

1. 605 条严格 OOF 预测上的整体 PR-AUC 高于 M0；
2. 在 5 个外层测试 fold 中至少 4 个 fold 的 PR-AUC 高于 M0。

若 M1、M2 都通过，则取整体 OOF PR-AUC 更高者；若均未通过，则保留 M0，遵循模型简约原则。

{md_table(select_display, ["task", "最终模型", "OOF_PR_AUC", "OOF_F1", "说明"], digits=3)}

完整候选比较见 `10_最终模型选择.csv`。

## 6. 孕妇级 Cluster Bootstrap 95% CI

置信区间以孕妇为重采样单位：每次有放回抽取 147 位孕妇，抽中某孕妇时带入她全部检测记录。共重复 {BOOT_REPS} 次。

{md_table(ci_display, list(ci_display.columns), digits=3)}

## 7. 敏感性分析

### 7.1 F2 阈值：更强调异常召回

正文主判定仍使用无主观风险权重的 F1 阈值。作为敏感性分析，在每个训练层内部改用 F2 选择阈值，模型和排序概率不变，从而观察提高召回率时 Precision/Specificity 的代价。

{md_table(selected_f2_display, ["task", "model", "threshold_policy", "Sensitivity", "Precision", "Specificity", "F1", "F2"], digits=3)}

### 7.2 X 染色体浓度是否有增益

M1 去掉 X 染色体浓度后，重新执行完整的外层 5-fold / 内层 3-fold 孕妇级嵌套验证，再与包含 X 浓度的 M1 比较 PR-AUC。

{md_table(x_sensitivity, list(x_sensitivity.columns), digits=3)}

### 7.3 GC、测序质量、孕周和 BMI 是否有增益

以 M1（染色体信息）和 M2（完整 16 特征）严格 OOF PR-AUC 的差值衡量新增变量的组外贡献。

{md_table(quality_sensitivity, list(quality_sensitivity.columns), digits=3)}

## 8. 最终全数据判定器

OOF 结果只用于报告泛化性能。模型结构确定后，再使用全部 147 位孕妇/605 条记录重新选择最终 C 与阈值并拟合操作型判定器。每条记录输出 T13/T18/T21 的概率/分数、阈值、0/1 判定，并自然组合成“无异常 / T13 / T18 / T21 / 复合异常”。

最终逐记录结果见 `12_全数据最终判定.csv`，最终参数见 `13_最终模型参数.csv`。若某任务最终保留 M0，则该任务分类仍由 $|Z_k|$ 阈值完成，同时额外给出仅用于 Brier/概率展示的单变量 Logistic 校准概率。

## 9. 检测质量标记

不把 GC 异常直接等同于胎儿异常，也不因 GC 越界删除记录。按题目给出的经验范围，只附加一个独立质量标记：

- $0.40 \\le GC \\le 0.60$：`GC范围内`；
- 否则：`GC范围外，可信度降低`。

因此“未判定异常”不等于“这次测序一定可靠”。

## 10. 图表

- `fig1_Z值阳性阴性分布.png`：三个目标的阳性/阴性 |Z| 分布；
- `fig2_PR曲线_M0_M1_M2.png`：严格 OOF PR 曲线；
- `fig3_最终模型校准曲线.png`：最终所选模型的 OOF 校准；T21 阳性极少，仅作描述；
- `fig4_Ridge标准化系数.png`：每个任务表现更好的 Ridge 系数方向。

## 11. 使用边界

**本文以附件 AB 列记录的 13、18、21 号染色体非整倍体检测结果作为监督标签，模型的目的在于学习和复现给定 NIPT 检测数据中的异常判读规律。由于女胎样本中的出生健康结局 AE 列缺乏类别变异，本文结果不能视为对胎儿真实染色体疾病或出生健康结局的独立临床诊断验证。**

## 12. 复现

在仓库根目录执行：

```bash
D:\\mypython\\python.exe 问题四\\code\\q4_pipeline.py
```

主要交付表：

1. `01_女胎数据结构.csv`
2. `02_特征定义.csv`
3. `03_单变量探索结果.csv`
4. `04_三层模型OOF结果.csv`
5. `05_最终性能及置信区间.csv`
6. `06_全模型cluster_bootstrap_CI.csv`
7. `07_外层fold分布.csv` / `07b_外层fold性能.csv`
8. `08_外层超参数.csv`
9. `09_OOF预测.csv`
10. `10_最终模型选择.csv`
11. `11_最终Ridge系数.csv`
12. `12_全数据最终判定.csv`
13. `13_最终模型参数.csv`
14. `14_F2阈值敏感性.csv`
15. `15_X染色体浓度敏感性.csv`
16. `16_测序质量变量敏感性.csv`
"""
(OUT / "问题四分析报告.md").write_text(report, encoding="utf-8")

summary = {
    "records": int(N),
    "women": int(df[GROUP_COL].nunique()),
    "positive_records": {task: int(Y[task].sum()) for task in TARGETS},
    "positive_women": {
        task: int(df.assign(_y=Y[task]).groupby(GROUP_COL)["_y"].max().sum())
        for task in TARGETS
    },
    "selected_model": selected_model,
    "selection_rule": f"overall PR-AUC > M0 and >= {MODEL_PASS_FOLD_WINS}/{OUTER_FOLDS} outer-fold PR-AUC wins",
    "selected_oof_metrics": {
        task: {
            m: safe_float(perf[(perf.task == task) & (perf.model == selected_model[task])][m].iloc[0])
            for m in ["Sensitivity", "Precision", "Specificity", "F1", "PR_AUC", "ROC_AUC", "Brier"]
        }
        for task in TARGETS
    },
    "bootstrap_reps": BOOT_REPS,
    "outer_folds": OUTER_FOLDS,
    "inner_folds": INNER_FOLDS,
    "random_seed": RANDOM_SEED,
}
(OUT / "运行摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("问题四流水线完成。")
print("最终模型：", selected_model)
print(select_display.to_string(index=False))
print("输出目录：", OUT)
