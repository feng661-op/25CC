"""2025 年 C 题 NIPT 数据的探索性分析、统计表和可视化。

请先运行：
    python src/preprocess.py
再运行：
    python src/eda.py
"""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_FILE = ROOT / "EDA_REPORT.md"

Y_CONCENTRATION = "Y染色体浓度"
WEEK = "孕周_连续值"
BMI = "BMI_calc"
PASS = "Y_pass"
ABNORMAL = "abnormal_any"

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
GROUP_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

LABELS = {
    "年龄": "年龄（岁）",
    "身高": "身高（cm）",
    "体重": "体重（kg）",
    "孕妇BMI": "原始 BMI",
    "BMI_calc": "BMI（按身高体重计算）",
    "BMI_diff": "BMI 差值",
    WEEK: "连续孕周（周）",
    "检测抽血次数": "检测抽血次数",
    "原始读段数": "原始读段数（条）",
    "在参考基因组上比对的比例": "比对比例",
    "重复读段的比例": "重复读段比例",
    "唯一比对的读段数": "唯一比对读段数（条）",
    "GC含量": "GC 含量（比例）",
    "13号染色体的Z值": "13 号染色体 Z 值",
    "18号染色体的Z值": "18 号染色体 Z 值",
    "21号染色体的Z值": "21 号染色体 Z 值",
    "X染色体的Z值": "X 染色体 Z 值",
    "Y染色体的Z值": "Y 染色体 Z 值",
    Y_CONCENTRATION: "Y 染色体浓度（比例值）",
    "X染色体浓度": "X 染色体浓度（比例值）",
    "13号染色体的GC含量": "13 号染色体 GC（比例）",
    "18号染色体的GC含量": "18 号染色体 GC（比例）",
    "21号染色体的GC含量": "21 号染色体 GC（比例）",
    "被过滤掉读段数的比例": "过滤读段比例",
    ABNORMAL: "任意异常（0/1）",
}

MALE_UNIVARIATE = [
    "年龄",
    "身高",
    "体重",
    BMI,
    WEEK,
    "检测抽血次数",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "GC含量",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "Y染色体的Z值",
    Y_CONCENTRATION,
    "X染色体浓度",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "被过滤掉读段数的比例",
]

FEMALE_UNIVARIATE = [
    "年龄",
    "身高",
    "体重",
    BMI,
    WEEK,
    "检测抽血次数",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "GC含量",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "X染色体浓度",
    "13号染色体的GC含量",
    "18号染色体的GC含量",
    "21号染色体的GC含量",
    "被过滤掉读段数的比例",
]

MALE_CORRELATION = [
    "年龄",
    "身高",
    "体重",
    BMI,
    WEEK,
    "检测抽血次数",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "GC含量",
    "被过滤掉读段数的比例",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "Y染色体的Z值",
    Y_CONCENTRATION,
    "X染色体浓度",
]

FEMALE_CORRELATION = [
    "年龄",
    "身高",
    "体重",
    BMI,
    WEEK,
    "检测抽血次数",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "唯一比对的读段数",
    "GC含量",
    "被过滤掉读段数的比例",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值",
    "X染色体的Z值",
    "X染色体浓度",
    ABNORMAL,
]

QUALITY_COLUMNS = [
    "GC含量",
    "原始读段数",
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "被过滤掉读段数的比例",
]


def configure_plot_style() -> str:
    """配置跨 Windows、macOS、Linux 的中文字体回退。"""
    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    selected = "DejaVu Sans"
    for candidate in font_candidates:
        try:
            fm.findfont(candidate, fallback_to_default=False)
            selected = candidate
            break
        except (ValueError, RuntimeError):
            continue

    sns.set_theme(style="ticks", context="notebook")
    sns.set_palette(OKABE_ITO)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    return selected


def load_processed(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到处理后数据：{path}，请先运行 python src/preprocess.py")
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["检测日期_日期", "末次月经_日期"])


def as_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def label(column: str) -> str:
    return LABELS.get(column, column)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_table(frame: pd.DataFrame, filename: str) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_p(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def add_lowess(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str = "#D55E00", label_text: str | None = "LOWESS趋势") -> None:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 10 or valid["x"].nunique() < 4:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = lowess(valid["y"], valid["x"], frac=0.45, it=0, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2.2, label=label_text or "_nolegend_", zorder=4)


def corr_pair(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    data = pd.concat([x, y], axis=1).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n < 3 or data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, n
    try:
        result = pearsonr(data.iloc[:, 0], data.iloc[:, 1]) if method == "pearson" else spearmanr(data.iloc[:, 0], data.iloc[:, 1])
        return float(result.statistic), float(result.pvalue), n
    except (ValueError, FloatingPointError):
        return np.nan, np.nan, n


def partial_spearman(df: pd.DataFrame, x_col: str, y_col: str, control_col: str) -> float:
    data = pd.DataFrame({"x": as_numeric(df, x_col), "y": as_numeric(df, y_col), "c": as_numeric(df, control_col)}).dropna()
    if len(data) < 5 or data[["x", "y", "c"]].nunique().min() < 2:
        return np.nan
    ranks = data.rank(method="average")
    design = np.column_stack([np.ones(len(ranks)), ranks["c"].to_numpy()])
    x_resid = ranks["x"].to_numpy() - design @ np.linalg.lstsq(design, ranks["x"].to_numpy(), rcond=None)[0]
    y_resid = ranks["y"].to_numpy() - design @ np.linalg.lstsq(design, ranks["y"].to_numpy(), rcond=None)[0]
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def compute_correlation_table(df: pd.DataFrame, columns: list[str], dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [column for column in columns if column in df.columns]
    rows = []
    numeric_data = pd.DataFrame({column: as_numeric(df, column) for column in columns})
    for left in columns:
        for right in columns:
            value, p_value, n = corr_pair(numeric_data[left], numeric_data[right], "spearman")
            rows.append({"数据集": dataset, "变量一": left, "变量二": right, "Spearman相关系数": value, "P值": p_value, "样本数": n})
    matrix = numeric_data.corr(method="spearman")
    matrix.index.name = "变量"
    return pd.DataFrame(rows), matrix


def compute_y_correlations(male: pd.DataFrame) -> pd.DataFrame:
    variables = [column for column in MALE_CORRELATION if column in male.columns and column != Y_CONCENTRATION]
    rows = []
    for column in variables:
        pr, pp, n_p = corr_pair(as_numeric(male, Y_CONCENTRATION), as_numeric(male, column), "pearson")
        sr, sp, n_s = corr_pair(as_numeric(male, Y_CONCENTRATION), as_numeric(male, column), "spearman")
        rows.append(
            {
                "变量": column,
                "变量名称": label(column),
                "Pearson相关系数": pr,
                "Pearson_P值": pp,
                "Spearman相关系数": sr,
                "Spearman_P值": sp,
                "样本数": min(n_p, n_s),
                "控制孕周后的部分Spearman": partial_spearman(male, Y_CONCENTRATION, column, WEEK) if column == BMI else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Spearman相关系数", key=lambda s: s.abs(), ascending=False)


def assign_bmi_groups(values: pd.Series, min_group_n: int = 12) -> tuple[pd.Series, pd.DataFrame, str]:
    """优先使用题目常用区间，组过小则自动切换到等频分组。"""
    numeric_values = pd.to_numeric(values, errors="coerce")
    fixed_bins = [-np.inf, 28, 32, 36, 40, np.inf]
    fixed_labels = ["BMI＜28", "28≤BMI＜32", "32≤BMI＜36", "36≤BMI＜40", "BMI≥40"]
    fixed = pd.cut(numeric_values, bins=fixed_bins, labels=fixed_labels, right=False)
    fixed_counts = fixed.value_counts().sort_index()
    nonempty = fixed_counts[fixed_counts > 0]
    if len(nonempty) >= 3 and int(nonempty.min()) >= min_group_n:
        definitions = pd.DataFrame(
            {
                "BMI组": fixed_labels,
                "分组方式": "经验区间（用于EDA）",
                "下界": [0, 28, 32, 36, 40],
                "上界": [28, 32, 36, 40, np.inf],
                "样本数": [int(fixed_counts.get(label_text, 0)) for label_text in fixed_labels],
            }
        )
        return fixed.astype("string"), definitions, "经验区间"

    valid = numeric_values.dropna()
    group_count = min(5, max(3, valid.nunique()))
    quantile_category = pd.qcut(valid, q=group_count, duplicates="drop")
    categories = list(quantile_category.cat.categories)
    quantile_labels = [f"等频组{i + 1}（{interval.left:.1f}–{interval.right:.1f}）" for i, interval in enumerate(categories)]
    mapped = pd.Series(pd.NA, index=values.index, dtype="string")
    mapping = {category: quantile_labels[i] for i, category in enumerate(categories)}
    mapped.loc[valid.index] = quantile_category.map(mapping).astype("string")
    counts = mapped.value_counts().reindex(quantile_labels, fill_value=0)
    definitions = pd.DataFrame(
        {
            "BMI组": quantile_labels,
            "分组方式": "等频分组（用于EDA）",
            "下界": [interval.left for interval in categories],
            "上界": [interval.right for interval in categories],
            "样本数": [int(counts.get(group, 0)) for group in quantile_labels],
        }
    )
    return mapped, definitions, "等频分组"


def plot_missing_rate(male: pd.DataFrame, female: pd.DataFrame) -> None:
    quality = pd.read_csv(TABLE_DIR / "missing_values.csv", encoding="utf-8-sig")
    quality = quality[quality["字段来源"] == "原始字段"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, dataset in zip(axes, ["男胎", "女胎"]):
        data = quality[quality["数据集"] == dataset].sort_values("缺失率", ascending=False).head(12)
        data = data[data["缺失数量"] > 0]
        if data.empty:
            ax.text(0.5, 0.5, "原始字段无缺失", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.barh(data["字段"].map(label)[::-1], data["缺失率"][::-1], color="#0072B2")
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.set_title(f"{dataset}（原始字段）")
        ax.set_xlabel("缺失率")
        ax.set_ylabel("")
    fig.suptitle("原始字段缺失率审计", y=1.02, fontsize=13)
    save_figure(fig, "01_missing_rate.png")


def plot_univariate_panels(df: pd.DataFrame, columns: list[str], dataset: str, filename: str) -> None:
    columns = [column for column in columns if column in df.columns]
    ncols = 3
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(14, max(3.0 * nrows, 6)), squeeze=False)
    axes_flat = axes.ravel()
    for i, column in enumerate(columns):
        values = as_numeric(df, column).dropna()
        hist_ax = axes_flat[i * 2]
        box_ax = axes_flat[i * 2 + 1]
        if values.empty:
            hist_ax.text(0.5, 0.5, "无非空值", ha="center", va="center")
            box_ax.set_axis_off()
            continue
        bins = min(30, max(8, int(np.sqrt(len(values)))))
        hist_ax.hist(values, bins=bins, color="#56B4E9", edgecolor="white", alpha=0.9)
        hist_ax.set_title(label(column), fontsize=9)
        hist_ax.set_ylabel("频数")
        hist_ax.set_xlabel(label(column))
        box_ax.boxplot(values, vert=False, patch_artist=True, boxprops={"facecolor": "#E69F00", "alpha": 0.65}, medianprops={"color": "#000000"})
        box_ax.set_title(f"箱线图（n={len(values)}）", fontsize=9)
        box_ax.set_xlabel(label(column))
        box_ax.set_yticks([])
    for ax in axes_flat[len(columns) * 2 :]:
        ax.set_visible(False)
    fig.suptitle(f"{dataset}关键数值变量分布总览（直方图与箱线图）", y=0.998, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    save_figure(fig, filename)


def plot_bmi_distribution(male: pd.DataFrame, female: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for data, name, color in [(male, "男胎", "#0072B2"), (female, "女胎", "#D55E00")]:
        values = as_numeric(data, BMI).dropna()
        ax.hist(values, bins=25, alpha=0.48, density=True, color=color, label=f"{name}（n={len(values)}）")
        if len(values) > 5:
            sns.kdeplot(values, ax=ax, color=color, linewidth=2)
    ax.set_xlabel("BMI（kg/m²，按身高体重计算）")
    ax.set_ylabel("密度")
    ax.set_title("BMI 分布")
    ax.legend(frameon=False)
    sns.despine()
    save_figure(fig, "02_bmi_distribution.png")


def plot_week_distribution(male: pd.DataFrame, female: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for data, name, color in [(male, "男胎", "#0072B2"), (female, "女胎", "#D55E00")]:
        values = as_numeric(data, WEEK).dropna()
        ax.hist(values, bins=np.arange(10.5, 30.6, 1), alpha=0.48, color=color, label=f"{name}（n={len(values)}）")
    ax.set_xlabel("连续孕周（周）")
    ax.set_ylabel("频数")
    ax.set_title("检测孕周分布")
    ax.legend(frameon=False)
    sns.despine()
    save_figure(fig, "03_gestational_week_distribution.png")


def plot_y_distribution(male: pd.DataFrame) -> None:
    values = as_numeric(male, Y_CONCENTRATION).dropna()
    pass_rate = float((values >= 0.04).mean())
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(values, bins=35, color="#56B4E9", alpha=0.85, edgecolor="white", density=True)
    if len(values) > 5:
        sns.kdeplot(values, ax=ax, color="#0072B2", linewidth=2.2, label="KDE趋势")
    ax.axvline(0.04, color="#D55E00", linestyle="--", linewidth=2, label="4%阈值（0.04）")
    ax.set_xlabel("Y 染色体浓度（比例值）")
    ax.set_ylabel("密度")
    ax.set_title(f"男胎 Y 染色体浓度分布：达标 {pass_rate:.1%}，未达标 {1 - pass_rate:.1%}")
    ax.legend(frameon=False)
    sns.despine()
    save_figure(fig, "04_y_distribution_threshold.png")


def plot_y_vs_week(male: pd.DataFrame) -> tuple[float, float, float, float]:
    y = as_numeric(male, Y_CONCENTRATION)
    week = as_numeric(male, WEEK)
    sr, sp, n_s = corr_pair(week, y, "spearman")
    pr, pp, _ = corr_pair(week, y, "pearson")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(week, y, s=17, alpha=0.35, color="#0072B2", edgecolors="none", label=f"原始观测（n={n_s}）")
    add_lowess(ax, week, y)
    ax.axhline(0.04, color="#D55E00", linestyle="--", linewidth=1.8, label="4%阈值（0.04）")
    ax.set_xlabel("连续孕周（周）")
    ax.set_ylabel("Y 染色体浓度（比例值）")
    ax.set_title("Y 染色体浓度与孕周：散点与非参数趋势")
    ax.text(0.02, 0.96, f"Spearman ρ={sr:.3f}（P={fmt_p(sp)}）\nPearson r={pr:.3f}（P={fmt_p(pp)}）", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(frameon=False)
    sns.despine()
    save_figure(fig, "05_y_vs_week.png")
    return sr, sp, pr, pp


def plot_y_vs_bmi(male: pd.DataFrame) -> tuple[float, float, float, float]:
    y = as_numeric(male, Y_CONCENTRATION)
    bmi = as_numeric(male, BMI)
    week = as_numeric(male, WEEK)
    sr, sp, _ = corr_pair(bmi, y, "spearman")
    pr, pp, _ = corr_pair(bmi, y, "pearson")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    valid = pd.DataFrame({"BMI": bmi, "Y": y, "week": week}).dropna()
    scatter = ax.scatter(valid["BMI"], valid["Y"], c=valid["week"], cmap="cividis", s=20, alpha=0.55, edgecolors="none")
    add_lowess(ax, bmi, y, color="#D55E00")
    ax.axhline(0.04, color="#0072B2", linestyle="--", linewidth=1.8, label="4%阈值（0.04）")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("连续孕周（周）")
    ax.set_xlabel("BMI（kg/m²，按身高体重计算）")
    ax.set_ylabel("Y 染色体浓度（比例值）")
    ax.set_title("Y 染色体浓度与 BMI：颜色表示孕周")
    ax.text(0.02, 0.96, f"Spearman ρ={sr:.3f}（P={fmt_p(sp)}）\nPearson r={pr:.3f}（P={fmt_p(pp)}）", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(frameon=False, loc="lower right")
    sns.despine()
    save_figure(fig, "06_y_vs_bmi.png")
    return sr, sp, pr, pp


def plot_week_bmi_heatmap(male: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame({"week": as_numeric(male, WEEK), "bmi": as_numeric(male, BMI), "pass": as_numeric(male, PASS)}).dropna()
    week_edges = list(np.arange(10, 32, 2))
    week_labels = [f"{left}–<{right}" for left, right in zip(week_edges[:-1], week_edges[1:])]
    bmi_edges = [-np.inf, 28, 32, 36, 40, np.inf]
    bmi_labels = ["BMI＜28", "28≤BMI＜32", "32≤BMI＜36", "36≤BMI＜40", "BMI≥40"]
    data["孕周分箱"] = pd.cut(data["week"], bins=week_edges, labels=week_labels, right=False)
    data["BMI分箱"] = pd.cut(data["bmi"], bins=bmi_edges, labels=bmi_labels, right=False)
    grouped = data.groupby(["BMI分箱", "孕周分箱"], observed=False)["pass"].agg(["mean", "count"]).reset_index()
    grouped["达标率"] = grouped["mean"]
    grouped["是否显示"] = grouped["count"] >= 5
    write_table(grouped.rename(columns={"mean": "达标率原值", "count": "样本数"}), "week_bmi_pass_heatmap.csv")

    rate = grouped.pivot(index="BMI分箱", columns="孕周分箱", values="达标率").reindex(index=bmi_labels, columns=week_labels)
    counts = grouped.pivot(index="BMI分箱", columns="孕周分箱", values="count").reindex(index=bmi_labels, columns=week_labels)
    displayed = rate.where(counts >= 5)
    annot = displayed.map(lambda value: "" if pd.isna(value) else f"{value:.0%}")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.heatmap(displayed, mask=displayed.isna(), cmap="viridis", vmin=0, vmax=1, annot=annot, fmt="", linewidths=0.5, linecolor="white", cbar_kws={"label": "Y≥4% 达标率"}, ax=ax)
    ax.set_xlabel("连续孕周分箱（周）")
    ax.set_ylabel("BMI 分箱")
    ax.set_title("孕周 × BMI 对 Y≥4% 达标率的二维探索（样本数＜5的格子不显示）")
    save_figure(fig, "07_week_bmi_pass_heatmap.png")
    return grouped.rename(columns={"mean": "达标率原值", "count": "样本数"})


def plot_pass_rate_by_bmi(male: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    data = pd.DataFrame({"week": as_numeric(male, WEEK), "bmi": as_numeric(male, BMI), "pass": as_numeric(male, PASS)}).dropna()
    data["BMI组"], definitions, method = assign_bmi_groups(data["bmi"], min_group_n=12)
    write_table(definitions, "bmi_group_definitions.csv")

    data["孕周整周"] = np.floor(data["week"]).astype(int)
    grouped = data.groupby(["BMI组", "孕周整周"], observed=False)["pass"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "达标率", "count": "样本数"})
    write_table(grouped, "pass_rate_by_bmi_week.csv")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    group_order = definitions["BMI组"].tolist()
    for i, group in enumerate(group_order):
        group_rows = data[data["BMI组"] == group]
        if group_rows.empty:
            continue
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        weekly = grouped[grouped["BMI组"] == group]
        ax.plot(weekly["孕周整周"], weekly["达标率"], marker="o", markersize=4, linewidth=1.2, color=color, alpha=0.8, label=f"{group}（n={len(group_rows)}）")
        add_lowess(ax, group_rows["week"], group_rows["pass"], color=color, label_text=None)
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("连续孕周（整周分箱）")
    ax.set_ylabel("Y≥4% 达标率")
    ax.set_title(f"不同 BMI 分组的 Y≥4% 达标率—孕周曲线（{method}）")
    ax.legend(frameon=False, ncol=2)
    sns.despine()
    save_figure(fig, "08_pass_rate_by_bmi.png")
    return grouped, definitions, method


def plot_trajectories(male: pd.DataFrame) -> pd.DataFrame:
    subject_summary = pd.read_csv(PROCESSED_DIR / "subject_summary_male.csv", encoding="utf-8-sig", parse_dates=["首次检测日期"])
    selected = subject_summary.sort_values(["记录数", "孕妇代码"], ascending=[False, True]).head(25)
    write_table(selected[["孕妇代码", "记录数", "不同检测孕周数", "BMI_首次记录", "首次达标孕周"]], "trajectory_selected_subjects.csv")

    selected_ids = set(selected["孕妇代码"].astype(str))
    data = male[male["孕妇代码"].astype(str).isin(selected_ids)].copy()
    data["week"] = as_numeric(data, WEEK)
    data["y"] = as_numeric(data, Y_CONCENTRATION)
    data = data.dropna(subset=["week", "y"])
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(1, data["孕妇代码"].nunique())))
    for color, (subject_id, group) in zip(colors, data.groupby("孕妇代码", sort=True)):
        group = group.sort_values("week")
        ax.plot(group["week"], group["y"], marker="o", markersize=3, linewidth=1.0, alpha=0.58, color=color)
    ax.axhline(0.04, color="#D55E00", linestyle="--", linewidth=2, label="4%阈值（0.04）")
    ax.set_xlabel("连续孕周（周）")
    ax.set_ylabel("Y 染色体浓度（比例值）")
    ax.set_title(f"记录次数最多的 {data['孕妇代码'].nunique()} 名孕妇 Y 浓度纵向轨迹")
    ax.text(0.02, 0.96, "每条线代表一名孕妇；用于展示个体内变化与个体间差异", transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(frameon=False)
    sns.despine()
    save_figure(fig, "09_repeated_measurement_trajectory.png")
    return selected


def plot_first_pass(male: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = pd.read_csv(PROCESSED_DIR / "subject_summary_male.csv", encoding="utf-8-sig", parse_dates=["首次检测日期"])
    first = subject.copy()
    first["是否观察到达标"] = 1 - pd.to_numeric(first["未观测达标"], errors="coerce").fillna(0).astype(int)
    write_table(first, "male_first_pass_week.csv")

    observed = first[pd.to_numeric(first["首次达标孕周"], errors="coerce").notna()].copy()
    never = int(first["未观测达标"].sum())
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    if not observed.empty:
        values = pd.to_numeric(observed["首次达标孕周"], errors="coerce")
        edges = np.arange(math.floor(values.min()) - 0.5, math.ceil(values.max()) + 1.5, 1)
        ax.hist(values, bins=edges, color="#009E73", alpha=0.85, edgecolor="white")
    ax.set_xlabel("首次观测达到 4% 的孕周（周）")
    ax.set_ylabel("孕妇人数")
    ax.set_title(f"首次观测达到 4% 的孕周分布（观察到达标 {len(observed)} 人；未观测到达标 {never} 人）")
    ax.text(0.98, 0.96, "首次观测即达标只能说明在该周已观测到达标，\n不能等同于真实生理达标时刻", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    sns.despine()
    save_figure(fig, "10_first_pass_week_distribution.png")

    group_values, definitions, method = assign_bmi_groups(first["BMI_首次记录"], min_group_n=8)
    first["BMI组"] = group_values
    pass_observed = first[first["首次达标孕周"].notna() & first["BMI组"].notna()].copy()
    pass_observed["首次达标孕周"] = pd.to_numeric(pass_observed["首次达标孕周"], errors="coerce")
    box_data = [pass_observed.loc[pass_observed["BMI组"] == group, "首次达标孕周"].dropna() for group in definitions["BMI组"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.05, 1.35]})
    axes[0].scatter(first.loc[first["首次达标孕周"].notna(), "BMI_首次记录"], first.loc[first["首次达标孕周"].notna(), "首次达标孕周"], s=25, alpha=0.7, color="#0072B2")
    axes[0].set_xlabel("BMI（首次记录，kg/m²）")
    axes[0].set_ylabel("首次观测达到 4% 的孕周（周）")
    axes[0].set_title("首次达标孕周与 BMI")
    axes[0].text(0.04, 0.96, f"观察到达标 n={len(pass_observed)}\n分组方式：{method}", transform=axes[0].transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    axes[1].boxplot(box_data, tick_labels=definitions["BMI组"].tolist(), patch_artist=True, boxprops={"facecolor": "#E69F00", "alpha": 0.65}, medianprops={"color": "#000000"}, showfliers=False)
    for i, values in enumerate(box_data, start=1):
        if len(values):
            jitter = np.random.default_rng(42).normal(i, 0.045, len(values))
            axes[1].scatter(jitter, values, s=12, alpha=0.45, color="#0072B2", zorder=3)
    axes[1].set_xlabel("BMI 分组（仅作 EDA）")
    axes[1].set_ylabel("首次观测达到 4% 的孕周（周）")
    axes[1].set_title("各 BMI 组首次达标孕周")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("BMI 与首次观测达到 4% 孕周的关系", y=1.02, fontsize=13)
    fig.tight_layout()
    sns.despine()
    save_figure(fig, "11_first_pass_week_vs_bmi.png")

    pass_table = first.groupby("BMI组", dropna=False, observed=False).agg(
        孕妇数=("孕妇代码", "size"),
        观察到达标人数=("是否观察到达标", "sum"),
        首次达标孕周中位数=("首次达标孕周", "median"),
        首次达标孕周Q1=("首次达标孕周", lambda s: s.quantile(0.25)),
        首次达标孕周Q3=("首次达标孕周", lambda s: s.quantile(0.75)),
        未观测达标比例=("未观测达标", "mean"),
    ).reset_index()
    pass_table["观察到达标比例"] = pass_table["观察到达标人数"] / pass_table["孕妇数"]
    write_table(pass_table, "first_pass_week_by_bmi.csv")
    return first, pass_table


def plot_male_correlation(male: pd.DataFrame) -> pd.DataFrame:
    table, matrix = compute_correlation_table(male, MALE_CORRELATION, "男胎")
    write_table(table, "male_spearman_correlation_long.csv")
    matrix.to_csv(TABLE_DIR / "male_spearman_correlation_matrix.csv", encoding="utf-8-sig")
    plot_matrix = matrix.rename(index=label, columns=label)
    mask = np.triu(np.ones_like(plot_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(plot_matrix, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=0.25, cbar_kws={"label": "Spearman 相关系数"}, ax=ax)
    ax.set_title("男胎候选变量 Spearman 相关性（下三角）")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=55, ha="right")
    plt.yticks(rotation=0)
    save_figure(fig, "12_male_spearman_heatmap.png")
    y_table = compute_y_correlations(male)
    write_table(y_table, "correlation_y.csv")
    return y_table


def plot_female_abnormal_distribution(female: pd.DataFrame) -> pd.DataFrame:
    order = ["正常", "T13", "T18", "T21", "复合异常"]
    counts = female["异常类型_分类"].value_counts().reindex(order, fill_value=0).rename_axis("异常类型").reset_index(name="数量")
    counts["比例"] = counts["数量"] / len(female)
    write_table(counts, "female_abnormal_counts.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(counts["异常类型"], counts["数量"], color=["#009E73", "#D55E00", "#CC79A7", "#E69F00", "#0072B2"])
    for bar, count in zip(bars, counts["数量"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts["数量"].max() * 0.015, 1), str(int(count)), ha="center", va="bottom")
    ax.set_xlabel("异常类型")
    ax.set_ylabel("记录数")
    ax.set_title(f"女胎异常类型频数（总记录 n={len(female)}；类别不平衡）")
    sns.despine()
    save_figure(fig, "13_female_abnormal_distribution.png")
    return counts


def rank_auc(y_true: pd.Series, score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true, "score": score}).dropna()
    if data["y"].nunique() < 2:
        return np.nan
    positive = data["y"].astype(int).to_numpy() == 1
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    ranks = rankdata(data["score"].to_numpy())
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compare_feature(df: pd.DataFrame, feature: str, target: str, subset_name: str) -> dict[str, object]:
    values = as_numeric(df, feature)
    target_values = as_numeric(df, target)
    data = pd.DataFrame({"value": values, "target": target_values}).dropna()
    negative = data.loc[data["target"] == 0, "value"]
    positive = data.loc[data["target"] == 1, "value"]
    if len(negative) and len(positive):
        u_result = mannwhitneyu(negative, positive, alternative="two-sided")
        p_value = float(u_result.pvalue)
        auc = rank_auc(data["target"], data["value"])
    else:
        p_value = np.nan
        auc = np.nan
    return {
        "比较": subset_name,
        "变量": feature,
        "变量名称": label(feature),
        "阴性样本数": int(len(negative)),
        "阳性样本数": int(len(positive)),
        "阴性中位数": float(negative.median()) if len(negative) else np.nan,
        "阳性中位数": float(positive.median()) if len(positive) else np.nan,
        "阴性Q1": float(negative.quantile(0.25)) if len(negative) else np.nan,
        "阴性Q3": float(negative.quantile(0.75)) if len(negative) else np.nan,
        "阳性Q1": float(positive.quantile(0.25)) if len(positive) else np.nan,
        "阳性Q3": float(positive.quantile(0.75)) if len(positive) else np.nan,
        "MannWhitney_P值": p_value,
        "方向性AUC": auc,
        "双侧判别AUC": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
    }


def plot_z_comparison(female: pd.DataFrame, feature: str, target: str, filename: str, title: str) -> dict[str, object]:
    target_values = as_numeric(female, target)
    feature_values = as_numeric(female, feature)
    subset = female.loc[(as_numeric(female, ABNORMAL) == 0) | (target_values == 1)].copy()
    subset["分组"] = np.where(as_numeric(subset, target) == 1, "对应异常（含复合）", "正常")
    subset["数值"] = as_numeric(subset, feature)
    subset = subset.dropna(subset=["数值"])
    comparison = compare_feature(subset.assign(目标=as_numeric(subset, target)), "数值", "目标", title)
    comparison["变量"] = feature
    comparison["变量名称"] = label(feature)
    order = ["正常", "对应异常（含复合）"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.boxplot(data=subset, x="分组", y="数值", order=order, hue="分组", palette=["#009E73", "#D55E00"], legend=False, ax=ax, showfliers=False)
    sns.stripplot(data=subset, x="分组", y="数值", order=order, color="#333333", alpha=0.35, size=2.5, jitter=0.22, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(label(feature))
    ax.set_title(title)
    ax.text(0.98, 0.96, f"阴性 n={int((subset['分组'] == order[0]).sum())}\n阳性 n={int((subset['分组'] == order[1]).sum())}", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.tick_params(axis="x", rotation=12)
    sns.despine()
    save_figure(fig, filename)
    return comparison


def plot_zx_comparison(female: pd.DataFrame) -> dict[str, object]:
    subset = female.copy()
    subset["分组"] = np.where(as_numeric(subset, ABNORMAL) == 1, "任意异常", "正常")
    subset["数值"] = as_numeric(subset, "X染色体的Z值")
    subset = subset.dropna(subset=["数值", "分组"])
    comparison = compare_feature(subset.assign(目标=(subset["分组"] == "任意异常").astype(int)), "数值", "目标", "ZX与任意异常")
    comparison["变量"] = "X染色体的Z值"
    comparison["变量名称"] = label("X染色体的Z值")
    order = ["正常", "任意异常"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.violinplot(data=subset, x="分组", y="数值", order=order, hue="分组", palette=["#009E73", "#D55E00"], legend=False, inner="quartile", cut=0, ax=ax)
    sns.stripplot(data=subset, x="分组", y="数值", order=order, color="#333333", alpha=0.35, size=2.5, jitter=0.22, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(label("X染色体的Z值"))
    ax.set_title("X 染色体 Z 值与任意异常")
    ax.text(0.98, 0.96, f"正常 n={int((subset['分组'] == '正常').sum())}\n异常 n={int((subset['分组'] == '任意异常').sum())}", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    sns.despine()
    save_figure(fig, "17_zx_vs_abnormal.png")
    return comparison


def plot_female_other_features(female: pd.DataFrame) -> pd.DataFrame:
    features = [BMI, WEEK, "GC含量", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例"]
    subset = female.copy()
    subset["分组"] = np.where(as_numeric(subset, ABNORMAL) == 1, "异常", "正常")
    rows = [compare_feature(subset, feature, ABNORMAL, "正常 vs 任意异常") for feature in features]
    summary = pd.DataFrame(rows)
    write_table(summary, "female_feature_abnormal_summary.csv")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax, feature in zip(axes.flat, features):
        plot_data = pd.DataFrame({"分组": subset["分组"], "数值": as_numeric(subset, feature)}).dropna()
        sns.boxplot(data=plot_data, x="分组", y="数值", order=["正常", "异常"], hue="分组", palette=["#009E73", "#D55E00"], legend=False, showfliers=False, ax=ax)
        sns.stripplot(data=plot_data, x="分组", y="数值", order=["正常", "异常"], color="#333333", alpha=0.22, size=1.8, jitter=0.2, ax=ax)
        ax.set_title(label(feature), fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)
    for ax in axes.flat[len(features) :]:
        ax.set_visible(False)
    fig.suptitle("女胎其他特征与任意异常的分布对比", y=1.0, fontsize=13)
    fig.tight_layout()
    save_figure(fig, "19_female_other_features_vs_abnormal.png")
    return summary


def plot_female_correlation(female: pd.DataFrame) -> None:
    table, matrix = compute_correlation_table(female, FEMALE_CORRELATION, "女胎")
    write_table(table, "female_spearman_correlation_long.csv")
    matrix.to_csv(TABLE_DIR / "female_spearman_correlation_matrix.csv", encoding="utf-8-sig")
    plot_matrix = matrix.rename(index=label, columns=label)
    mask = np.triu(np.ones_like(plot_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(plot_matrix, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=0.25, cbar_kws={"label": "Spearman 相关系数"}, ax=ax)
    ax.set_title("女胎候选变量 Spearman 相关性（不含 Y 相关变量）")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=55, ha="right")
    plt.yticks(rotation=0)
    save_figure(fig, "18_female_spearman_heatmap.png")


def plot_quality_vs_y(male: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in QUALITY_COLUMNS:
        sr, sp, n = corr_pair(as_numeric(male, feature), as_numeric(male, Y_CONCENTRATION), "spearman")
        group = pd.DataFrame({"quality": as_numeric(male, feature), "pass": as_numeric(male, PASS)}).dropna()
        pass_median = group.loc[group["pass"] == 1, "quality"].median()
        fail_median = group.loc[group["pass"] == 0, "quality"].median()
        rows.append({"测序质量变量": feature, "变量名称": label(feature), "与Y浓度Spearman": sr, "P值": sp, "样本数": n, "达标组中位数": pass_median, "未达标组中位数": fail_median})
    summary = pd.DataFrame(rows)
    write_table(summary, "quality_vs_y_summary.csv")

    fig, axes = plt.subplots(2, len(QUALITY_COLUMNS), figsize=(18, 7.2))
    for i, feature in enumerate(QUALITY_COLUMNS):
        quality = as_numeric(male, feature)
        y = as_numeric(male, Y_CONCENTRATION)
        axes[0, i].scatter(quality, y, s=11, alpha=0.3, color="#0072B2", edgecolors="none")
        add_lowess(axes[0, i], quality, y, color="#D55E00", label_text=None)
        axes[0, i].axhline(0.04, color="#777777", linestyle="--", linewidth=1)
        axes[0, i].set_xlabel(label(feature), fontsize=8)
        axes[0, i].set_ylabel("Y浓度" if i == 0 else "")
        axes[0, i].set_title(f"ρ={summary.loc[i, '与Y浓度Spearman']:.2f}", fontsize=9)
        plot_data = pd.DataFrame({"质量": quality, "达标": np.where(as_numeric(male, PASS) == 1, "达标", "未达标")}).dropna()
        sns.boxplot(data=plot_data, x="达标", y="质量", order=["未达标", "达标"], hue="达标", palette=["#D55E00", "#009E73"], legend=False, showfliers=False, ax=axes[1, i])
        axes[1, i].set_xlabel("")
        axes[1, i].set_ylabel(label(feature) if i == 0 else "")
        axes[1, i].tick_params(axis="x", rotation=25)
        if feature == "原始读段数":
            axes[0, i].set_xscale("log")
            axes[1, i].set_yscale("log")
    fig.suptitle("测序质量变量与 Y 浓度及 4% 达标状态", y=1.0, fontsize=13)
    fig.tight_layout()
    save_figure(fig, "19_quality_vs_y.png")
    return summary


def make_near_week_pairs(male: pd.DataFrame, max_week_diff: float = 0.5) -> pd.DataFrame:
    rows = []
    for subject_id, group in male.groupby("孕妇代码", sort=True):
        group = group[["孕妇代码", "序号", WEEK, Y_CONCENTRATION]].copy()
        group[WEEK] = as_numeric(group, WEEK)
        group[Y_CONCENTRATION] = as_numeric(group, Y_CONCENTRATION)
        group = group.dropna(subset=[WEEK, Y_CONCENTRATION]).sort_values(WEEK)
        for left, right in itertools.combinations(group.to_dict("records"), 2):
            week_diff = abs(float(right[WEEK] - left[WEEK]))
            if week_diff <= max_week_diff:
                y_mean = (float(left[Y_CONCENTRATION]) + float(right[Y_CONCENTRATION])) / 2
                y_diff = float(right[Y_CONCENTRATION] - left[Y_CONCENTRATION])
                rows.append(
                    {
                        "孕妇代码": subject_id,
                        "记录一": left["序号"],
                        "记录二": right["序号"],
                        "孕周差": week_diff,
                        "Y浓度均值": y_mean,
                        "Y浓度差（后者-前者）": y_diff,
                        "Y浓度绝对差": abs(y_diff),
                    }
                )
    return pd.DataFrame(rows)


def plot_repeat_measurement_difference(male: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = pd.read_csv(PROCESSED_DIR / "subject_summary_male.csv", encoding="utf-8-sig")
    variability = subject[subject["记录数"] >= 2].copy()
    variability_table = variability[["孕妇代码", "记录数", "Y浓度标准差", "Y浓度极差", "BMI_首次记录"]].copy()
    write_table(variability_table, "male_within_subject_variability.csv")
    pairs = make_near_week_pairs(male, max_week_diff=0.5)
    write_table(pairs, "near_week_repeat_pairs.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    axes[0].hist(pd.to_numeric(variability["Y浓度标准差"], errors="coerce").dropna(), bins=25, color="#56B4E9", edgecolor="white")
    axes[0].set_xlabel("同一孕妇 Y 浓度标准差")
    axes[0].set_ylabel("孕妇人数")
    axes[0].set_title(f"个体内标准差（n={len(variability)}）")
    axes[1].hist(pd.to_numeric(variability["Y浓度极差"], errors="coerce").dropna(), bins=25, color="#E69F00", edgecolor="white")
    axes[1].set_xlabel("同一孕妇 Y 浓度极差")
    axes[1].set_ylabel("孕妇人数")
    axes[1].set_title("个体内极差")
    if pairs.empty:
        axes[2].text(0.5, 0.5, "没有足够的近似同孕周重复对\n（阈值：孕周差≤0.5周）", ha="center", va="center")
        axes[2].set_axis_off()
    else:
        axes[2].scatter(pairs["Y浓度均值"], pairs["Y浓度差（后者-前者）"], s=20, alpha=0.55, color="#0072B2")
        axes[2].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[2].set_xlabel("两次 Y 浓度均值")
        axes[2].set_ylabel("后次 − 前次 Y 浓度")
        axes[2].set_title(f"近孕周重复差值（n={len(pairs)}）")
    fig.suptitle("男胎重复测量的个体内波动", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, "20_repeat_measurement_difference.png")
    return variability, pairs


def compute_subject_slopes(male: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, group in male.groupby("孕妇代码", sort=True):
        data = pd.DataFrame({"week": as_numeric(group, WEEK), "y": as_numeric(group, Y_CONCENTRATION)}).dropna()
        if len(data) < 2 or data["week"].nunique() < 2:
            continue
        slope, intercept = np.polyfit(data["week"], data["y"], 1)
        rows.append({"孕妇代码": subject_id, "记录数": len(data), "Y浓度对孕周线性斜率": slope, "正斜率": int(slope > 0)})
    result = pd.DataFrame(rows)
    write_table(result, "male_subject_slopes.csv")
    return result


def create_health_consistency_table(male: pd.DataFrame, female: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, df in [("男胎", male), ("女胎", female)]:
        cross = pd.crosstab(df["异常类型_分类"], df["胎儿是否健康"], dropna=False)
        for abnormal_type, row in cross.iterrows():
            for health, count in row.items():
                rows.append({"数据集": dataset, "异常类型": abnormal_type, "健康状态": health, "数量": int(count)})
    result = pd.DataFrame(rows)
    write_table(result, "health_abnormal_consistency.csv")
    return result


def report_link(filename: str) -> str:
    return f"[查看图表](outputs/figures/{filename})"


def build_report(
    male: pd.DataFrame,
    female: pd.DataFrame,
    male_y_corr: pd.DataFrame,
    first_pass: pd.DataFrame,
    first_pass_by_bmi: pd.DataFrame,
    pass_rate_by_bmi: pd.DataFrame,
    bmi_method: str,
    female_counts: pd.DataFrame,
    female_discrimination: pd.DataFrame,
    female_other: pd.DataFrame,
    quality_y: pd.DataFrame,
    variability: pd.DataFrame,
    pairs: pd.DataFrame,
    slopes: pd.DataFrame,
    health_consistency: pd.DataFrame,
    font_name: str,
) -> None:
    male_subject = pd.read_csv(PROCESSED_DIR / "subject_summary_male.csv", encoding="utf-8-sig")
    female_subject = pd.read_csv(PROCESSED_DIR / "subject_summary_female.csv", encoding="utf-8-sig")
    male_y = as_numeric(male, Y_CONCENTRATION)
    male_week = as_numeric(male, WEEK)
    male_bmi = as_numeric(male, BMI)
    pass_rate = float((male_y >= 0.04).mean())
    y_week_sr, y_week_sp, _ = corr_pair(male_week, male_y, "spearman")
    y_bmi_sr, y_bmi_sp, _ = corr_pair(male_bmi, male_y, "spearman")
    y_bmi_partial = partial_spearman(male, Y_CONCENTRATION, BMI, WEEK)
    first_pass_observed = first_pass[first_pass["首次达标孕周"].notna()]
    never_count = int(first_pass["未观测达标"].sum())
    repeated_subject_count = int(male_subject["是否重复测量"].sum())
    repeated_rate = repeated_subject_count / len(male_subject)
    positive_slope_rate = float(slopes["正斜率"].mean()) if not slopes.empty else np.nan
    abnormal_total = int((female[ABNORMAL] == 1).sum())
    abnormal_rate = abnormal_total / len(female)
    male_qc_top = quality_y.sort_values("与Y浓度Spearman", key=lambda s: s.abs(), ascending=False).iloc[0]
    female_auc = female_discrimination.sort_values("双侧判别AUC", ascending=False).head(4)
    female_corr = female_other.sort_values("双侧判别AUC", ascending=False).head(4)
    z_rows = female_discrimination[female_discrimination["比较"].isin(["T13异常", "T18异常", "T21异常"])]
    z_auc_min = float(z_rows["双侧判别AUC"].min()) if not z_rows.empty else np.nan
    z_auc_max = float(z_rows["双侧判别AUC"].max()) if not z_rows.empty else np.nan

    manifest = pd.read_csv(TABLE_DIR / "data_manifest.csv", encoding="utf-8-sig")
    source_hash = manifest["原始文件SHA256"].iloc[0]
    source_path = manifest["原始文件"].iloc[0]
    male_counts = male_subject["记录数"].value_counts().sort_index()
    female_counts_subject = female_subject["记录数"].value_counts().sort_index()

    bmi_lines = []
    for _, row in first_pass_by_bmi.dropna(subset=["BMI组"]).iterrows():
        bmi_lines.append(
            f"- {row['BMI组']}：{int(row['孕妇数'])} 人，观察到达标 {int(row['观察到达标人数'])} 人（{fmt_pct(row['观察到达标比例'])}）；"
            f"首次达标孕周中位数 {fmt(row['首次达标孕周中位数'], 2)} 周，未观测达标比例 {fmt_pct(row['未观测达标比例'])}。"
        )

    top_y_lines = []
    for _, row in male_y_corr.head(8).iterrows():
        top_y_lines.append(f"- {row['变量名称']}：Spearman ρ={fmt(row['Spearman相关系数'])}（P={fmt_p(row['Spearman_P值'])}，n={int(row['样本数'])}）。")

    discrimination_lines = []
    for _, row in female_auc.iterrows():
        discrimination_lines.append(f"- {row['变量名称']}（{row['比较']}）：双侧判别 AUC={fmt(row['双侧判别AUC'])}，P={fmt_p(row['MannWhitney_P值'])}。")

    other_lines = []
    for _, row in female_corr.iterrows():
        other_lines.append(f"- {row['变量名称']}：正常中位数 {fmt(row['阴性中位数'])}，异常中位数 {fmt(row['阳性中位数'])}，双侧判别 AUC={fmt(row['双侧判别AUC'])}。")

    report = f"""# 2025 C题 NIPT 数据探索性分析

## 1 数据概况

本次分析优先读取仓库内的 2025 年 C 题原始附件 `{source_path}`；该文件与公开来源中的原始附件同 SHA-256：
<https://github.com/luluzzy/CUMCM2025-C-Problem/blob/main/files/%E9%99%84%E4%BB%B6.xlsx>

本次分析使用原始 Excel 两张工作表，文件 SHA-256 为 `{source_hash}`。分析代码为 `src/preprocess.py` 和 `src/eda.py`，没有使用其他仓库已经清洗好的男胎或女胎 CSV。

| 数据集 | 记录行数 | 孕妇人数 | 重复测量孕妇 | 重复测量比例 |
|---|---:|---:|---:|---:|
| 男胎 | {len(male)} | {male['孕妇代码'].nunique()} | {repeated_subject_count} | {fmt_pct(repeated_rate)} |
| 女胎 | {len(female)} | {female['孕妇代码'].nunique()} | {int(female_subject['是否重复测量'].sum())} | {fmt_pct(float(female_subject['是否重复测量'].mean()))} |

男胎每名孕妇记录数为 {int(male_subject['记录数'].min())}–{int(male_subject['记录数'].max())} 条，中位数 {fmt(male_subject['记录数'].median(), 0)} 条；女胎为 {int(female_subject['记录数'].min())}–{int(female_subject['记录数'].max())} 条，中位数 {fmt(female_subject['记录数'].median(), 0)} 条。男胎记录数分布见 `outputs/tables/subject_summary_male.csv`，女胎对应表见 `outputs/tables/subject_summary_female.csv`。

## 2 数据质量

### 2.1 字段保留与标准化

- 列名只去除了首尾空格，原始字段和值均保留；处理后数据另行追加连续孕周、计算 BMI、BMI 差值、4% 达标标志、异常拆分和重复测量信息。
- 孕周按“周数 + 天数 / 7”换算。例如 `11w+6` 被换算为 `11.857142...`，没有把 6 天写成 0.6 周。
- 检测日期和末次月经分别增加了可计算的日期字段，但原始日期字段仍保留。

### 2.2 缺失、结构性缺失与一致性

- 处理后男胎表（含追加字段）缺失单元格共 {int(male.isna().sum().sum())} 个；原始字段中主要是末次月经 {int(male['末次月经'].isna().sum())} 条和异常类型空白 {int(male['染色体的非整倍体'].isna().sum())} 条。
- 处理后女胎表（含追加字段）缺失单元格共 {int(female.isna().sum().sum())} 个；女胎的 Y 染色体 Z 值和 Y 染色体浓度为空是生理机制导致的结构性缺失，不能当作质量问题，不能填 0，也不能用均值填补。
- 男胎 BMI 重新计算后，BMI 差值绝对值中位数为 {fmt(pd.to_numeric(male['BMI_diff'], errors='coerce').abs().median(), 6)}；女胎为 {fmt(pd.to_numeric(female['BMI_diff'], errors='coerce').abs().median(), 6)}。差值接近 0，说明原始 BMI 与身高、体重整体一致；女胎有 {int(female['孕妇BMI'].isna().sum())} 条原始 BMI 缺失，但仍可由身高体重计算 BMI。
- GC 按约 0.40–0.60 做范围标记，不删除范围外记录。男胎被标记 {int(male['GC_abnormal'].sum())} 行，女胎被标记 {int(female['GC_abnormal'].sum())} 行；这类标记不等于测序失败，需要结合质量变量继续判断。
- 所有比例变量均检查是否在 0–1；检查结果见 `outputs/tables/proportion_range_check.csv`。孕周检查范围为 8–42 周，解析失败和范围异常均只标记。
- 异常值采用标志方式：BMI、GC、Y 浓度、原始读段数同时给出 IQR 1.5 倍规则标志，未按 IQR 或 Z 分数批量删除。完整标志表见 `outputs/tables/outlier_flags_summary.csv`。

缺失率图：{report_link('01_missing_rate.png')}；BMI 图：{report_link('02_bmi_distribution.png')}；孕周图：{report_link('03_gestational_week_distribution.png')}。

### 2.3 重复检测结构

男胎存在 {repeated_subject_count} 名重复测量孕妇，占孕妇人数 {fmt_pct(repeated_rate)}；女胎存在 {int(female_subject['是否重复测量'].sum())} 名重复测量孕妇。完全重复行、同孕妇同日期、同孕妇同孕周和同次采血重复检测的审计结果见 `outputs/tables/duplicate_audit.csv`。

这不是完全独立样本数据：同一孕妇在不同孕周重复出现，且记录次数并不完全相同。后续模型应考虑个体随机效应、按孕妇分组交叉验证或其他能处理受试者内相关性的办法。

## 3 男胎主要规律

### 3.1 Y 浓度与 4% 阈值

男胎 Y 浓度的观测值范围为 {fmt(male_y.min(), 4)}–{fmt(male_y.max(), 4)}；按 Y≥0.04（4%）定义，{len(male)} 条记录中达标 {int((male_y >= 0.04).sum())} 条（{fmt_pct(pass_rate)}），未达标 {int((male_y < 0.04).sum())} 条（{fmt_pct(1-pass_rate)}）。

{report_link('04_y_distribution_threshold.png')}

### 3.2 Y 浓度与孕周

Y 浓度与连续孕周的 Spearman 相关系数为 {fmt(y_week_sr)}（P={fmt_p(y_week_sp)}），说明当前记录中总体呈正向单调关系；Pearson 结果见 `outputs/tables/correlation_y.csv`。散点和 LOWESS 趋势显示关系并非只应由一条直线概括，早期低浓度区和后期高浓度区的离散程度不同。

因此，当前数据支持“孕周增加时，Y 浓度总体倾向升高”的探索性结论，但不能把观测相关直接解释成每名孕妇都按相同速度增长。{report_link('05_y_vs_week.png')}

### 3.3 Y 浓度与 BMI

Y 浓度与计算 BMI 的 Spearman 相关系数为 {fmt(y_bmi_sr)}（P={fmt_p(y_bmi_sp)}）；控制连续孕周后的探索性部分 Spearman 相关为 {fmt(y_bmi_partial)}。BMI 图使用颜色标出了孕周，目的就是检查“BMI 表面相关、实际受孕周混杂”的可能性。

当前结果应表述为：高 BMI 区域的 Y 浓度和较早孕周达标率可能偏低，但 BMI 与孕周、个体差异及测序质量同时存在关系，不能写成“BMI 越大一定越晚”。{report_link('06_y_vs_bmi.png')}

### 3.4 孕周 × BMI 对达标率

二维热力图每个格子展示的是该孕周和 BMI 区间内的 Y≥4% 达标率，不是样本数量；样本数小于 5 的格子不显示。孕周分为每 2 周一档，BMI 使用 28、32、36、40 作为探索性边界。分箱只用于 EDA，不代表后续问题二的最终分组方案。{report_link('07_week_bmi_pass_heatmap.png')}

不同 BMI 组的达标率—孕周曲线使用“{bmi_method}”，具体边界和每组样本量见 `outputs/tables/bmi_group_definitions.csv`。{report_link('08_pass_rate_by_bmi.png')}

BMI 组与首次观测达标孕周的概览如下：

{chr(10).join(bmi_lines)}

这些结果可用于提出后续建模假设：BMI 可能影响达到 4% 的观测时点，但应在孕周、多次测量和质量变量共同进入模型后再判断其独立作用。

### 3.5 重复测量与个体差异

在至少有 2 条男胎记录的孕妇中，Y 浓度个体内标准差的中位数为 {fmt(variability['Y浓度标准差'].median(), 4)}，个体内极差中位数为 {fmt(variability['Y浓度极差'].median(), 4)}。基于每名孕妇内部“Y 浓度—孕周”简单线性斜率的 {len(slopes)} 名可估计孕妇中，正斜率比例为 {fmt_pct(positive_slope_rate)}；这只是描述性指标，不是最终模型。

记录次数最多的孕妇纵向轨迹见 {report_link('09_repeated_measurement_trajectory.png')}；首次观测达到 4% 的孕周分布见 {report_link('10_first_pass_week_distribution.png')}，BMI 对应图见 {report_link('11_first_pass_week_vs_bmi.png')}。

首次达标孕周存在左截断：如果第一次检测已经达到 4%，只能说“首次观测时已达标”，不能声称真实生理达标时刻就是该周；从未观测达到 4% 的 {never_count} 名孕妇必须单独保留，不能为作图而删除或补值。

### 3.6 测序质量与 Y 结果

与 Y 浓度绝对 Spearman 相关最大的质量变量是 {male_qc_top['变量名称']}，相关系数为 {fmt(male_qc_top['与Y浓度Spearman'])}（P={fmt_p(male_qc_top['P值'])}）。完整质量比较表见 `outputs/tables/quality_vs_y_summary.csv`，图中同时展示 Y 浓度散点、趋势和按达标状态的质量分布：{report_link('19_quality_vs_y.png')}。

这说明后续不能把低 Y 浓度全部视为纯生理现象；GC、原始读段数、比对比例、重复比例和过滤比例应作为质量协变量或敏感性分析变量。

Y 浓度与候选变量的相关系数排序如下，优先参考 Spearman：

{chr(10).join(top_y_lines)}

## 4 女胎主要规律

### 4.1 异常类型与类别不平衡

女胎共 {len(female)} 条记录，其中任意异常 {abnormal_total} 条（{fmt_pct(abnormal_rate)}），正常 {len(female)-abnormal_total} 条。具体类型为：{'；'.join([f"{row['异常类型']} {int(row['数量'])} 条" for _, row in female_counts.iterrows()])}。复合异常单独归类，T13、T18、T21 指示变量同时保留。

{report_link('13_female_abnormal_distribution.png')}

    """
    report = report.rstrip() + "\n\n"
    report += "### 4.2 染色体 Z 值与对应异常\n\n"
    report += "箱线图按‘正常’与‘对应异常（含复合异常）’比较；其他不包含该染色体的异常记录不混入对应阳性组。"
    report += "\n\n" + "\n".join(discrimination_lines) + "\n\n"
    report += f"{report_link('14_z13_vs_t13.png')}　{report_link('15_z18_vs_t18.png')}　{report_link('16_z21_vs_t21.png')}\n\n"
    report += f"X 染色体 Z 值与任意异常的比较：{report_link('17_zx_vs_abnormal.png')}。"
    report += "\n\n### 4.3 BMI、孕周和测序质量与异常\n\n"
    report += "在当前女胎记录中，其他特征与任意异常的分布差异按双侧判别 AUC 排序，前几项为：\n\n" + "\n".join(other_lines) + "\n\n"
    report += f"其他特征对比图：{report_link('19_female_other_features_vs_abnormal.png')}；女胎候选变量 Spearman 热力图（不含 Y 相关变量）：{report_link('18_female_spearman_heatmap.png')}。"
    report += f"\n\n当前附件中，三个对应染色体 Z 值的双侧判别 AUC 约为 {fmt(z_auc_min)}–{fmt(z_auc_max)}，没有出现明显的组间分离；它们仍是机制上最直接的候选特征，但需要核对标签定义并通过分层交叉验证确认，当前图表不能替代正式诊断性能评估。"
    report += "\n\n## 5 给后续建模人员的建议\n\n"
    report += "1. **数据结构**：以孕妇为分组单位处理重复测量；训练、验证和交叉验证不要把同一孕妇拆到不同折中。\n"
    report += "2. **孕周与 BMI**：把连续孕周保留为连续变量；BMI 分箱图只作为可解释的 EDA 结果，不要直接把经验边界当成最终最优分组。可比较达标概率曲线、首次观测达标孕周和不确定性。\n"
    report += "3. **Y 达标相关问题**：同时考虑孕周、BMI、个体层级差异和测序质量；首次观测达标具有左截断，未达标者不能简单删除。\n"
    report += "4. **女胎异常判定**：13、18、21 号染色体的 Z 值是最直接的候选变量，同时评估 X 染色体 Z 值、BMI、孕周和测序质量；复合异常应保留多标签信息，不能只留一个字符串类别。\n"
    report += "5. **类别不平衡**：报告分层召回率、特异度、精确率、PR 曲线或其他适合少数类的指标，不只看总体准确率。\n"
    report += "6. **避免信息泄漏**：‘胎儿是否健康’属于后验结果，只能做描述或一致性检查，不能作为异常预测输入。当前一致性表见 `outputs/tables/health_abnormal_consistency.csv`。\n"
    report += "7. **异常值与敏感性分析**：先使用标志字段和质量分层，必要时再做有明确规则的敏感性分析；不要把所有 IQR 外记录自动删除。\n\n"
    report += "## 6 可复现文件清单\n\n"
    report += "- 处理脚本：`src/preprocess.py`、`src/eda.py`。\n"
    report += "- 处理数据：`data/processed/male_cleaned.csv`、`data/processed/female_cleaned.csv`。\n"
    report += "- 质量审计：`outputs/tables/data_quality_summary.csv`、`missing_values.csv`、`duplicate_audit.csv`、`proportion_range_check.csv`、`bmi_consistency_summary.csv`、`outlier_flags_summary.csv`。\n"
    report += "- 纵向与达标分析：`male_first_pass_week.csv`、`first_pass_week_by_bmi.csv`、`male_within_subject_variability.csv`、`near_week_repeat_pairs.csv`。\n"
    report += "- 相关性与质量：`correlation_y.csv`、`male_spearman_correlation_matrix.csv`、`female_spearman_correlation_matrix.csv`、`quality_vs_y_summary.csv`。\n"
    report += f"- 所有图表采用统一风格，PNG 分辨率为 300 dpi；本次运行使用的中文字体回退为：{font_name}。\n"
    report += "\n> 本报告是数据审计和 EDA 结果，不等同于真实临床诊断结论，也不等同于后续四问模型的最终验收。"
    report = report.rstrip()

    REPORT_FILE.write_text(report, encoding="utf-8")


def main() -> None:
    font_name = configure_plot_style()
    male = load_processed("male_cleaned.csv")
    female = load_processed("female_cleaned.csv")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_missing_rate(male, female)
    plot_bmi_distribution(male, female)
    plot_week_distribution(male, female)
    plot_univariate_panels(male, MALE_UNIVARIATE, "男胎", "21_male_univariate_panels.png")
    plot_univariate_panels(female, FEMALE_UNIVARIATE, "女胎", "22_female_univariate_panels.png")
    plot_y_distribution(male)
    y_week = plot_y_vs_week(male)
    y_bmi = plot_y_vs_bmi(male)
    plot_week_bmi_heatmap(male)
    pass_rate_bmi, _, bmi_method = plot_pass_rate_by_bmi(male)
    plot_trajectories(male)
    first_pass, first_pass_by_bmi = plot_first_pass(male)
    male_y_corr = plot_male_correlation(male)
    female_counts = plot_female_abnormal_distribution(female)

    discrimination_rows = []
    target_specs = [
        ("13号染色体的Z值", "abnormal_T13", "T13异常"),
        ("18号染色体的Z值", "abnormal_T18", "T18异常"),
        ("21号染色体的Z值", "abnormal_T21", "T21异常"),
    ]
    for feature, target, title in target_specs:
        subset = female.loc[(as_numeric(female, ABNORMAL) == 0) | (as_numeric(female, target) == 1)].copy()
        comparison = compare_feature(subset, feature, target, title)
        discrimination_rows.append(comparison)
        filename = {"T13异常": "14_z13_vs_t13.png", "T18异常": "15_z18_vs_t18.png", "T21异常": "16_z21_vs_t21.png"}[title]
        plot_z_comparison(female, feature, target, filename, f"{label(feature)}与{title}")
    zx_comparison = plot_zx_comparison(female)
    discrimination_rows.append(zx_comparison)
    female_discrimination = pd.DataFrame(discrimination_rows)
    write_table(female_discrimination, "female_discrimination_summary.csv")
    female_other = plot_female_other_features(female)
    plot_female_correlation(female)
    quality_y = plot_quality_vs_y(male)
    variability, pairs = plot_repeat_measurement_difference(male)
    slopes = compute_subject_slopes(male)
    health_consistency = create_health_consistency_table(male, female)

    build_report(
        male,
        female,
        male_y_corr,
        first_pass,
        first_pass_by_bmi,
        pass_rate_bmi,
        bmi_method,
        female_counts,
        female_discrimination,
        female_other,
        quality_y,
        variability,
        pairs,
        slopes,
        health_consistency,
        font_name,
    )
    print(f"EDA 完成：生成 {len(list(FIGURE_DIR.glob('*.png')))} 张 PNG 图和报告 {REPORT_FILE.name}。")


if __name__ == "__main__":
    main()
