"""2025 年 C 题 NIPT 数据的探索性分析、统计表和可视化。

请先运行：
    python src/preprocess.py
再运行：
    python src/eda.py
"""

from __future__ import annotations

import itertools
import math
import re
import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
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
MEASUREMENT_BMI = "measurement_BMI"
BASELINE_BMI = "baseline_BMI"
PASS = "Y_pass"
ABNORMAL = "abnormal_any"

BOOTSTRAP_SEED = 42
BOOTSTRAP_ITERATIONS = 2000
SLOPE_NEAR_ZERO_THRESHOLD = 0.0005

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
GROUP_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

LABELS = {
    "年龄": "年龄（岁）",
    "身高": "身高（cm）",
    "体重": "体重（kg）",
    "孕妇BMI": "原始 BMI",
    "BMI_calc": "BMI（按身高体重计算）",
    MEASUREMENT_BMI: "检测时 BMI（kg/m²）",
    BASELINE_BMI: "基线 BMI（最早检测孕周）",
    "BMI_diff": "BMI 差值",
    WEEK: "连续孕周（周）",
    "检测抽血次数": "检测抽血次数",
    "原始读段数": "原始读段数（条）",
    "在参考基因组上比对的比例": "比对比例",
    "重复读段的比例": "重复读段比例",
    "唯一比对的读段数": "唯一比对读段数（条）",
    "GC含量": "GC 含量（比例）",
    "GC_range_flag": "GC 经验范围标记",
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
    MEASUREMENT_BMI,
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
    MEASUREMENT_BMI,
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
    MEASUREMENT_BMI,
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
    MEASUREMENT_BMI,
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

FINAL_FIGURE_NAMES = [
    "01_missing_rate.png",
    "02_bmi_distribution.png",
    "03_gestational_week_distribution.png",
    "04_y_distribution_threshold.png",
    "05_y_vs_week.png",
    "06_y_vs_measurement_bmi.png",
    "07_week_baseline_bmi_pass_heatmap.png",
    "08_pass_rate_by_baseline_bmi.png",
    "09_repeated_measurement_trajectory.png",
    "10_first_observed_pass_week_distribution.png",
    "11_first_observed_pass_week_vs_baseline_bmi.png",
    "12_male_spearman_heatmap.png",
    "13_male_subject_slope_distribution.png",
    "14_male_threshold_censoring_types.png",
    "15_male_same_draw_repeat_error.png",
    "16_quality_vs_y.png",
    "17_female_abnormal_distribution.png",
    "18_z13_vs_t13.png",
    "19_z18_vs_t18.png",
    "20_z21_vs_t21.png",
    "21_zx_vs_abnormal.png",
    "22_female_spearman_heatmap.png",
    "23_female_other_features_vs_abnormal.png",
    "24_female_subject_abnormal_summary.png",
    "25_male_univariate_panels.png",
    "26_female_univariate_panels.png",
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
            rows.append(
                {
                    "数据集": dataset,
                    "变量一": left,
                    "变量二": right,
                    "record_level_Spearman_rho": value,
                    "record_level_Spearman_p_naive": p_value,
                    "record_level_n": n,
                }
            )
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
                "record_level_Pearson_r": pr,
                "record_level_Pearson_p_naive": pp,
                "record_level_Spearman_rho": sr,
                "record_level_Spearman_p_naive": sp,
                "record_level_n": min(n_p, n_s),
                "partial_Spearman_control_week": partial_spearman(male, Y_CONCENTRATION, column, WEEK)
                if column == MEASUREMENT_BMI
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("record_level_Spearman_rho", key=lambda s: s.abs(), ascending=False)


def cluster_bootstrap_statistic(
    df: pd.DataFrame,
    statistic,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """按孕妇整簇重采样，重复抽中的孕妇作为独立 bootstrap 簇副本。"""
    subjects = sorted(df["孕妇代码"].dropna().astype(str).unique())
    groups = [df[df["孕妇代码"].astype(str) == subject].copy() for subject in subjects]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        selected = rng.integers(0, len(groups), size=len(groups))
        sampled = pd.concat([groups[index] for index in selected], ignore_index=True)
        try:
            value = float(statistic(sampled))
        except (ValueError, FloatingPointError, TypeError):
            value = np.nan
        values.append(value)
    return np.asarray(values, dtype="float64")


def partial_spearman_arrays(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    x = x[valid]
    y = y[valid]
    control = control[valid]
    if len(x) < 5 or min(np.unique(x).size, np.unique(y).size, np.unique(control).size) < 2:
        return np.nan
    ranks_x = rankdata(x, method="average")
    ranks_y = rankdata(y, method="average")
    ranks_control = rankdata(control, method="average")
    design = np.column_stack([np.ones(len(ranks_control)), ranks_control])
    x_resid = ranks_x - design @ np.linalg.lstsq(design, ranks_x, rcond=None)[0]
    y_resid = ranks_y - design @ np.linalg.lstsq(design, ranks_y, rcond=None)[0]
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def fast_cluster_bootstrap(male: pd.DataFrame, metric: str) -> np.ndarray:
    """用 NumPy 数组完成同一整簇 bootstrap，避免每次重建 DataFrame。"""
    subjects = sorted(male["孕妇代码"].dropna().astype(str).unique())
    subject_values = []
    for subject in subjects:
        group = male[male["孕妇代码"].astype(str) == subject]
        subject_values.append(
            (
                as_numeric(group, WEEK).to_numpy(dtype="float64"),
                as_numeric(group, MEASUREMENT_BMI).to_numpy(dtype="float64"),
                as_numeric(group, Y_CONCENTRATION).to_numpy(dtype="float64"),
            )
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        selected = rng.integers(0, len(subject_values), size=len(subject_values))
        weeks = np.concatenate([subject_values[index][0] for index in selected])
        bmis = np.concatenate([subject_values[index][1] for index in selected])
        ys = np.concatenate([subject_values[index][2] for index in selected])
        if metric == "Y浓度 vs 孕周":
            valid = np.isfinite(weeks) & np.isfinite(ys)
            value = spearmanr(weeks[valid], ys[valid]).statistic if valid.sum() >= 3 else np.nan
        elif metric == "Y浓度 vs measurement_BMI":
            valid = np.isfinite(bmis) & np.isfinite(ys)
            value = spearmanr(bmis[valid], ys[valid]).statistic if valid.sum() >= 3 else np.nan
        else:
            value = partial_spearman_arrays(ys, bmis, weeks)
        results.append(float(value) if np.isfinite(value) else np.nan)
    return np.asarray(results, dtype="float64")


def cluster_bootstrap_correlations(male: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "Y浓度 vs 孕周",
            lambda data: corr_pair(as_numeric(data, WEEK), as_numeric(data, Y_CONCENTRATION), "spearman")[0],
        ),
        (
            "Y浓度 vs measurement_BMI",
            lambda data: corr_pair(as_numeric(data, MEASUREMENT_BMI), as_numeric(data, Y_CONCENTRATION), "spearman")[0],
        ),
        (
            "Y浓度 vs measurement_BMI（控制孕周）",
            lambda data: partial_spearman(data, Y_CONCENTRATION, MEASUREMENT_BMI, WEEK),
        ),
    ]
    rows = []
    for metric, statistic in specs:
        point = float(statistic(male))
        boot = fast_cluster_bootstrap(male, metric)
        valid = boot[np.isfinite(boot)]
        rows.append(
            {
                "指标": metric,
                "record_level_point_estimate": point,
                "cluster_bootstrap_median": float(np.median(valid)) if len(valid) else np.nan,
                "CI95_lower": float(np.percentile(valid, 2.5)) if len(valid) else np.nan,
                "CI95_upper": float(np.percentile(valid, 97.5)) if len(valid) else np.nan,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "bootstrap_valid_iterations": int(len(valid)),
                "unique_subjects": int(male["孕妇代码"].nunique()),
                "record_level_n": int(
                    corr_pair(as_numeric(male, WEEK), as_numeric(male, Y_CONCENTRATION), "spearman")[2]
                    if metric == "Y浓度 vs 孕周"
                    else corr_pair(as_numeric(male, MEASUREMENT_BMI), as_numeric(male, Y_CONCENTRATION), "spearman")[2]
                ),
            }
        )
    result = pd.DataFrame(rows)
    write_table(result, "cluster_bootstrap_correlations.csv")
    return result


def assign_bmi_groups(values: pd.Series, min_group_n: int = 15) -> tuple[pd.Series, pd.DataFrame, str]:
    """按孕妇级 baseline BMI 分组；经验组过小则自动切换为等频分组。"""
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
                "下界": [-np.inf, 28, 32, 36, 40],
                "上界": [28, 32, 36, 40, np.inf],
                "孕妇人数": [int(fixed_counts.get(label_text, 0)) for label_text in fixed_labels],
            }
        )
        return fixed.astype("string"), definitions, "经验区间"

    valid = numeric_values.dropna()
    if valid.empty:
        definitions = pd.DataFrame(columns=["BMI组", "分组方式", "下界", "上界", "孕妇人数"])
        return pd.Series(pd.NA, index=values.index, dtype="string"), definitions, "无有效 baseline BMI"
    group_count = min(5, max(3, int(valid.size)))
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
            "孕妇人数": [int(counts.get(group, 0)) for group in quantile_labels],
        }
    )
    return mapped, definitions, "等频分组"


def prepare_bmi_grouping(male: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """建立唯一的孕妇级 BMI 分组，并映射回记录层数据。"""
    subject = male[["孕妇代码", BASELINE_BMI]].drop_duplicates("孕妇代码").copy()
    unique_counts = male.groupby("孕妇代码", dropna=False)[BASELINE_BMI].nunique(dropna=True)
    assert unique_counts.le(1).all(), "同一孕妇存在多个 baseline_BMI"
    groups, definitions, method = assign_bmi_groups(subject[BASELINE_BMI])
    subject["BMI组"] = groups.to_numpy()
    grouped = male.merge(subject[["孕妇代码", "BMI组"]], on="孕妇代码", how="left", validate="many_to_one")
    record_counts = grouped.groupby("BMI组", dropna=False)["孕妇代码"].size()
    subject_counts = subject.groupby("BMI组", dropna=False)["孕妇代码"].nunique()
    definitions["孕妇人数"] = [int(subject_counts.get(group, 0)) for group in definitions["BMI组"]]
    definitions["记录数"] = [int(record_counts.get(group, 0)) for group in definitions["BMI组"]]
    definitions = definitions[["BMI组", "分组方式", "下界", "上界", "孕妇人数", "记录数"]]
    write_table(definitions, "bmi_group_definitions.csv")
    assert grouped.groupby("孕妇代码")["BMI组"].nunique(dropna=True).le(1).all()
    return grouped, subject, definitions, method


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
        box_ax.set_title(f"箱线图（n={len(values)} records）", fontsize=9)
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
        ax.hist(values, bins=25, alpha=0.48, density=True, color=color, label=f"{name}（n={len(values)} records）")
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
        ax.hist(values, bins=np.arange(10.5, 30.6, 1), alpha=0.48, color=color, label=f"{name}（n={len(values)} records）")
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
    ax.set_title(f"男胎 Y 染色体浓度分布（n={len(values)} records）：达标 {pass_rate:.1%}，未达标 {1 - pass_rate:.1%}")
    ax.legend(frameon=False)
    sns.despine()
    save_figure(fig, "04_y_distribution_threshold.png")


def plot_y_vs_week(male: pd.DataFrame, bootstrap_row: pd.Series | None = None) -> tuple[float, float, float, float]:
    y = as_numeric(male, Y_CONCENTRATION)
    week = as_numeric(male, WEEK)
    sr, sp, n_s = corr_pair(week, y, "spearman")
    pr, pp, _ = corr_pair(week, y, "pearson")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(week, y, s=17, alpha=0.35, color="#0072B2", edgecolors="none", label=f"原始观测（n={n_s} records）")
    add_lowess(ax, week, y)
    ax.axhline(0.04, color="#D55E00", linestyle="--", linewidth=1.8, label="4%阈值（0.04）")
    ax.set_xlabel("连续孕周（周）")
    ax.set_ylabel("Y 染色体浓度（比例值）")
    ax.set_title("Y 染色体浓度与孕周：散点与非参数趋势")
    annotation = f"记录层面 Spearman ρ={sr:.3f}（普通探索性 P={fmt_p(sp)}）\n记录层面 Pearson r={pr:.3f}（普通探索性 P={fmt_p(pp)}）"
    if bootstrap_row is not None:
        annotation += f"\n按孕妇聚类 bootstrap 95% CI=[{fmt(bootstrap_row['CI95_lower'])}, {fmt(bootstrap_row['CI95_upper'])}]"
    ax.text(0.02, 0.96, annotation, transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(frameon=False, loc="lower right")
    sns.despine()
    save_figure(fig, "05_y_vs_week.png")
    return sr, sp, pr, pp


def plot_y_vs_bmi(male: pd.DataFrame, bootstrap_row: pd.Series | None = None) -> tuple[float, float, float, float]:
    y = as_numeric(male, Y_CONCENTRATION)
    bmi = as_numeric(male, MEASUREMENT_BMI)
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
    ax.set_xlabel("检测时 BMI / measurement-time BMI（kg/m²）")
    ax.set_ylabel("Y 染色体浓度（比例值）")
    ax.set_title("Y 染色体浓度与检测时 BMI：颜色表示孕周")
    annotation = f"记录层面 Spearman ρ={sr:.3f}（普通探索性 P={fmt_p(sp)}）\n记录层面 Pearson r={pr:.3f}（普通探索性 P={fmt_p(pp)}）"
    if bootstrap_row is not None:
        annotation += f"\n按孕妇聚类 bootstrap 95% CI=[{fmt(bootstrap_row['CI95_lower'])}, {fmt(bootstrap_row['CI95_upper'])}]"
    ax.text(0.02, 0.96, annotation, transform=ax.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(frameon=False, loc="lower right")
    sns.despine()
    save_figure(fig, "06_y_vs_measurement_bmi.png")
    return sr, sp, pr, pp


def plot_week_bmi_heatmap(male: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "week": as_numeric(male, WEEK),
            "BMI组": male["BMI组"].astype("string"),
            "pass": as_numeric(male, PASS),
            "孕妇代码": male["孕妇代码"],
        }
    ).dropna(subset=["week", "BMI组", "pass"])
    week_edges = list(np.arange(10, 32, 2))
    week_labels = [f"{left}–<{right}" for left, right in zip(week_edges[:-1], week_edges[1:])]
    data["孕周分箱"] = pd.cut(data["week"], bins=week_edges, labels=week_labels, right=False)
    bmi_labels = definitions["BMI组"].tolist()
    data["BMI组"] = pd.Categorical(data["BMI组"], categories=bmi_labels, ordered=True)
    grouped = (
        data.groupby(["BMI组", "孕周分箱"], observed=False)
        .agg(达标率=("pass", "mean"), 记录数=("pass", "count"), 孕妇数=("孕妇代码", "nunique"))
        .reset_index()
    )
    grouped["是否显示"] = grouped["记录数"] >= 5
    grouped["达标率原值"] = grouped["达标率"]
    write_table(grouped, "week_bmi_pass_heatmap.csv")

    rate = grouped.pivot(index="BMI组", columns="孕周分箱", values="达标率").reindex(index=bmi_labels, columns=week_labels)
    counts = grouped.pivot(index="BMI组", columns="孕周分箱", values="记录数").reindex(index=bmi_labels, columns=week_labels)
    displayed = rate.where(counts >= 5)
    annot = displayed.map(lambda value: "" if pd.isna(value) else f"{value:.0%}")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.heatmap(displayed, mask=displayed.isna(), cmap="viridis", vmin=0, vmax=1, annot=annot, fmt="", linewidths=0.5, linecolor="white", cbar_kws={"label": "Y≥4% 达标率"}, ax=ax)
    ax.set_xlabel("连续孕周分箱（周）")
    ax.set_ylabel("孕妇级 baseline BMI 分组")
    ax.set_title("孕周 × baseline BMI 对 Y≥4% 达标率的二维探索（记录数＜5的格子不显示）")
    save_figure(fig, "07_week_baseline_bmi_pass_heatmap.png")
    return grouped


def plot_pass_rate_by_bmi(male: pd.DataFrame, definitions: pd.DataFrame, method: str) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "week": as_numeric(male, WEEK),
            "BMI组": male["BMI组"].astype("string"),
            "pass": as_numeric(male, PASS),
            "孕妇代码": male["孕妇代码"],
        }
    ).dropna(subset=["week", "BMI组", "pass"])
    data["孕周整周"] = np.floor(data["week"]).astype(int)
    grouped = (
        data.groupby(["BMI组", "孕周整周"], observed=False)
        .agg(达标率=("pass", "mean"), 记录数=("pass", "count"), 孕妇数=("孕妇代码", "nunique"))
        .reset_index()
    )
    write_table(grouped, "pass_rate_by_bmi_week.csv")

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    group_order = definitions["BMI组"].tolist()
    for i, group in enumerate(group_order):
        group_rows = data[data["BMI组"] == group]
        if group_rows.empty:
            continue
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        weekly = grouped[grouped["BMI组"] == group]
        subject_n = int(definitions.loc[definitions["BMI组"] == group, "孕妇人数"].iloc[0])
        ax.plot(weekly["孕周整周"], weekly["达标率"], marker="o", markersize=4, linewidth=1.2, color=color, alpha=0.8, label=f"{group}（n={subject_n} subjects）")
        add_lowess(ax, group_rows["week"], group_rows["pass"], color=color, label_text=None)
    ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("连续孕周（整周分箱）")
    ax.set_ylabel("Y≥4% 达标率")
    ax.set_title(f"不同 baseline BMI 分组的 Y≥4% 达标率—孕周曲线（{method}）")
    ax.legend(frameon=False, ncol=2)
    sns.despine()
    save_figure(fig, "08_pass_rate_by_baseline_bmi.png")
    return grouped


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


def plot_first_pass(threshold: pd.DataFrame, definitions: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = threshold.copy()
    first["是否观察到达标"] = 1 - pd.to_numeric(first["从未观测达标"], errors="coerce").fillna(0).astype(int)
    write_table(first, "male_first_pass_week.csv")

    observed = first[pd.to_numeric(first["首次观测达标孕周"], errors="coerce").notna()].copy()
    never = int(first["从未观测达标"].sum())
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    if not observed.empty:
        values = pd.to_numeric(observed["首次观测达标孕周"], errors="coerce")
        edges = np.arange(math.floor(values.min()) - 0.5, math.ceil(values.max()) + 1.5, 1)
        ax.hist(values, bins=edges, color="#009E73", alpha=0.85, edgecolor="white")
    ax.set_xlabel("首次观测到 Y≥4% 的孕周（周）")
    ax.set_ylabel("孕妇人数")
    ax.set_title(f"首次观测到 Y≥4% 的孕周分布（观察到达标 {len(observed)} 人；未观测到达标 {never} 人）")
    ax.text(0.98, 0.96, "首次观测达标存在观测删失，\n不能等同于真实阈值跨越时刻", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    sns.despine()
    save_figure(fig, "10_first_observed_pass_week_distribution.png")

    pass_observed = first[first["首次观测达标孕周"].notna() & first["BMI组"].notna()].copy()
    pass_observed["首次观测达标孕周"] = pd.to_numeric(pass_observed["首次观测达标孕周"], errors="coerce")
    box_data = [pass_observed.loc[pass_observed["BMI组"] == group, "首次观测达标孕周"].dropna() for group in definitions["BMI组"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.05, 1.35]})
    scatter_data = first[first["首次观测达标孕周"].notna() & first[BASELINE_BMI].notna()]
    axes[0].scatter(scatter_data[BASELINE_BMI], scatter_data["首次观测达标孕周"], s=25, alpha=0.7, color="#0072B2")
    axes[0].set_xlabel("baseline BMI（kg/m²，最早检测孕周）")
    axes[0].set_ylabel("首次观测到 Y≥4% 的孕周（周）")
    axes[0].set_title("首次观测达标孕周与 baseline BMI")
    axes[0].text(0.04, 0.96, f"观察到达标 n={len(pass_observed)} subjects\n分组方式：{method}", transform=axes[0].transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    axes[1].boxplot(box_data, tick_labels=definitions["BMI组"].tolist(), patch_artist=True, boxprops={"facecolor": "#E69F00", "alpha": 0.65}, medianprops={"color": "#000000"}, showfliers=False)
    for i, values in enumerate(box_data, start=1):
        if len(values):
            jitter = np.random.default_rng(BOOTSTRAP_SEED).normal(i, 0.045, len(values))
            axes[1].scatter(jitter, values, s=12, alpha=0.45, color="#0072B2", zorder=3)
    axes[1].set_xlabel("baseline BMI 分组（仅作 EDA）")
    axes[1].set_ylabel("首次观测到 Y≥4% 的孕周（周）")
    axes[1].set_title("各 baseline BMI 组首次观测达标孕周")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("baseline BMI 与首次观测到 Y≥4% 孕周的关系", y=1.02, fontsize=13)
    fig.tight_layout()
    sns.despine()
    save_figure(fig, "11_first_observed_pass_week_vs_baseline_bmi.png")

    pass_table = first.groupby("BMI组", dropna=False, observed=False).agg(
        孕妇数=("孕妇代码", "nunique"),
        观察到达标人数=("是否观察到达标", "sum"),
        首次观测达标孕周中位数=("首次观测达标孕周", "median"),
        首次观测达标孕周Q1=("首次观测达标孕周", lambda s: s.quantile(0.25)),
        首次观测达标孕周Q3=("首次观测达标孕周", lambda s: s.quantile(0.75)),
        未观测达标比例=("从未观测达标", "mean"),
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
    ax.set_title(f"女胎异常类型频数（总记录 n={len(female)} records；类别不平衡）")
    sns.despine()
    save_figure(fig, "17_female_abnormal_distribution.png")
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
        auc_raw = rank_auc(data["target"], data["value"])
        auc_abs = rank_auc(data["target"], data["value"].abs())
    else:
        p_value = np.nan
        auc_raw = np.nan
        auc_abs = np.nan
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
        "P值说明": "record-level exploratory P value",
        "AUC_raw": auc_raw,
        "AUC_abs": auc_abs,
    }


def plot_z_comparison(female: pd.DataFrame, feature: str, target: str, filename: str, title: str) -> dict[str, object]:
    target_values = as_numeric(female, target)
    feature_values = as_numeric(female, feature)
    subset = female.loc[(as_numeric(female, ABNORMAL) == 0) | (target_values == 1)].copy()
    subset["分组"] = np.where(as_numeric(subset, target) == 1, "对应异常（含复合）", "正常")
    subset["数值"] = as_numeric(subset, feature)
    subset = subset.dropna(subset=["数值"])
    comparison = compare_feature(subset.assign(目标=as_numeric(subset, target)), "数值", "目标", title)
    comparison["AUC_raw_Z"] = comparison.pop("AUC_raw")
    comparison["AUC_abs_Z"] = comparison.pop("AUC_abs")
    comparison["变量"] = feature
    comparison["变量名称"] = label(feature)
    order = ["正常", "对应异常（含复合）"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.boxplot(data=subset, x="分组", y="数值", order=order, hue="分组", palette=["#009E73", "#D55E00"], legend=False, ax=ax, showfliers=False)
    sns.stripplot(data=subset, x="分组", y="数值", order=order, color="#333333", alpha=0.35, size=2.5, jitter=0.22, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(label(feature))
    ax.set_title(title)
    ax.text(0.98, 0.96, f"阴性 n={int((subset['分组'] == order[0]).sum())} records\n阳性 n={int((subset['分组'] == order[1]).sum())} records", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
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
    comparison["AUC_raw_Z"] = comparison.pop("AUC_raw")
    comparison["AUC_abs_Z"] = comparison.pop("AUC_abs")
    comparison["变量"] = "X染色体的Z值"
    comparison["变量名称"] = label("X染色体的Z值")
    order = ["正常", "任意异常"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.violinplot(data=subset, x="分组", y="数值", order=order, hue="分组", palette=["#009E73", "#D55E00"], legend=False, inner="quartile", cut=0, ax=ax)
    sns.stripplot(data=subset, x="分组", y="数值", order=order, color="#333333", alpha=0.35, size=2.5, jitter=0.22, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(label("X染色体的Z值"))
    ax.set_title("X 染色体 Z 值与任意异常")
    ax.text(0.98, 0.96, f"正常 n={int((subset['分组'] == '正常').sum())} records\n异常 n={int((subset['分组'] == '任意异常').sum())} records", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    sns.despine()
    save_figure(fig, "21_zx_vs_abnormal.png")
    return comparison


def plot_female_other_features(female: pd.DataFrame) -> pd.DataFrame:
    features = [MEASUREMENT_BMI, WEEK, "GC含量", "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例"]
    subset = female.copy()
    subset["分组"] = np.where(as_numeric(subset, ABNORMAL) == 1, "异常", "正常")
    rows = [compare_feature(subset, feature, ABNORMAL, "正常 vs 任意异常") for feature in features]
    summary = pd.DataFrame(rows)
    write_table(summary, "female_other_feature_abnormal_summary.csv")
    legacy_summary = summary.copy()
    legacy_summary.insert(0, "表格状态", "legacy_compatibility")
    write_table(legacy_summary, "female_feature_abnormal_summary.csv")

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
    fig.tight_layout()
    fig.suptitle("女胎其他特征与任意异常的记录层面分布对比（n = records）", y=1.0, fontsize=13)
    save_figure(fig, "23_female_other_features_vs_abnormal.png")
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
    save_figure(fig, "22_female_spearman_heatmap.png")


def plot_quality_vs_y(male: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in QUALITY_COLUMNS:
        sr, sp, n = corr_pair(as_numeric(male, feature), as_numeric(male, Y_CONCENTRATION), "spearman")
        group = pd.DataFrame({"quality": as_numeric(male, feature), "pass": as_numeric(male, PASS)}).dropna()
        pass_median = group.loc[group["pass"] == 1, "quality"].median()
        fail_median = group.loc[group["pass"] == 0, "quality"].median()
        rows.append(
            {
                "测序质量变量": feature,
                "变量名称": label(feature),
                "record_level_Spearman_rho": sr,
                "record_level_Spearman_p_naive": sp,
                "record_level_n": n,
                "P值说明": "record-level exploratory P value",
                "达标组中位数": pass_median,
                "未达标组中位数": fail_median,
            }
        )
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
        axes[0, i].set_title(f"ρ={summary.loc[i, 'record_level_Spearman_rho']:.2f}", fontsize=9)
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
    save_figure(fig, "16_quality_vs_y.png")
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


def make_threshold_censoring(male: pd.DataFrame) -> pd.DataFrame:
    """为每名男胎孕妇构造 4% 阈值的观测删失区间。"""
    rows = []
    for subject_id, group in male.groupby("孕妇代码", sort=True):
        ordered = group.copy()
        ordered["_week"] = as_numeric(ordered, WEEK)
        ordered["_y"] = as_numeric(ordered, Y_CONCENTRATION)
        ordered = ordered.dropna(subset=["_week", "_y"]).sort_values(
            ["_week", "检测日期_日期", "序号"], na_position="last", kind="mergesort"
        )
        baseline = as_numeric(group, BASELINE_BMI).dropna()
        bmi_group = group["BMI组"].dropna().astype(str).iloc[0] if "BMI组" in group and group["BMI组"].notna().any() else np.nan
        row = {
            "孕妇代码": subject_id,
            BASELINE_BMI: float(baseline.iloc[0]) if not baseline.empty else np.nan,
            "BMI组": bmi_group,
            "记录数": int(len(group)),
            "有效Y记录数": int(len(ordered)),
            "最早观测孕周": float(ordered["_week"].min()) if not ordered.empty else np.nan,
            "最晚观测孕周": float(ordered["_week"].max()) if not ordered.empty else np.nan,
            "首次观测达标孕周": np.nan,
            "最后一次达标前未达标孕周": np.nan,
            "是否观察到达标": 0,
            "删失类型": "right",
            "threshold_lower": np.nan,
            "threshold_upper": np.nan,
            "区间宽度": np.nan,
            "首次观测即达标": 0,
            "从未观测达标": 1,
            "threshold_nonmonotonic": 0,
            "阈值轨迹非单调": 0,
        }
        if not ordered.empty:
            pass_mask = ordered["_y"].ge(0.04).to_numpy()
            pass_positions = np.flatnonzero(pass_mask)
            if len(pass_positions):
                first_position = int(pass_positions[0])
                first_week = float(ordered.iloc[first_position]["_week"])
                row.update(
                    {
                        "首次观测达标孕周": first_week,
                        "是否观察到达标": 1,
                        "从未观测达标": 0,
                        "首次观测即达标": int(first_position == 0),
                        "threshold_nonmonotonic": int(ordered.iloc[first_position + 1 :]["_y"].lt(0.04).any()),
                        "阈值轨迹非单调": int(ordered.iloc[first_position + 1 :]["_y"].lt(0.04).any()),
                    }
                )
                if first_position == 0:
                    row["删失类型"] = "left"
                    row["threshold_upper"] = first_week
                else:
                    below_before = ordered.iloc[:first_position].loc[ordered.iloc[:first_position]["_y"] < 0.04]
                    lower = float(below_before["_week"].max()) if not below_before.empty else np.nan
                    row.update(
                        {
                            "删失类型": "interval",
                            "最后一次达标前未达标孕周": lower,
                            "threshold_lower": lower,
                            "threshold_upper": first_week,
                            "区间宽度": first_week - lower if np.isfinite(lower) else np.nan,
                        }
                    )
            else:
                row["threshold_lower"] = float(ordered["_week"].max())
        rows.append(row)
    result = pd.DataFrame(rows)
    result["censoring_type"] = result["删失类型"]
    result["阈值区间下界"] = result["threshold_lower"]
    result["阈值区间上界"] = result["threshold_upper"]
    write_table(result, "male_threshold_censoring.csv")
    assert len(result) == male["孕妇代码"].nunique()
    assert result["孕妇代码"].nunique() == len(result)
    assert set(result["删失类型"].dropna()) <= {"left", "interval", "right"}
    return result


def summarize_threshold_censoring(threshold: pd.DataFrame) -> pd.DataFrame:
    denominator = len(threshold)
    rows = []
    for name, count, description in [
        ("left", int((threshold["删失类型"] == "left").sum()), "第一次有效观测已达到阈值，只能知道 T 不晚于首次观测孕周"),
        ("interval", int((threshold["删失类型"] == "interval").sum()), "最后一次未达标与首次达标之间的观测区间"),
        ("right", int((threshold["删失类型"] == "right").sum()), "截至最后一次观测仍未达到阈值"),
        ("nonmonotonic", int(threshold["threshold_nonmonotonic"].sum()), "跨过阈值后又出现低于阈值的观测；与删失类型不互斥"),
    ]:
        rows.append({"项目": name, "数量": count, "比例": count / denominator if denominator else np.nan, "分母孕妇数": denominator, "说明": description})
    result = pd.DataFrame(rows)
    write_table(result, "threshold_censoring_summary.csv")
    return result


def summarize_threshold_censoring_by_bmi(threshold: pd.DataFrame) -> pd.DataFrame:
    data = threshold.dropna(subset=["BMI组"]).copy()
    rows = []
    for bmi_group, group in data.groupby("BMI组", sort=False):
        denominator = len(group)
        for censoring_type in ["left", "interval", "right"]:
            count = int((group["删失类型"] == censoring_type).sum())
            rows.append(
                {
                    "BMI组": bmi_group,
                    "删失类型": censoring_type,
                    "孕妇数": count,
                    "孕妇比例": count / denominator if denominator else np.nan,
                    "分组孕妇数": denominator,
                    "分组依据": "baseline_BMI（孕妇级）",
                }
            )
    result = pd.DataFrame(rows)
    write_table(result, "threshold_censoring_by_bmi.csv")
    return result


def plot_threshold_censoring_types(summary: pd.DataFrame, by_bmi: pd.DataFrame) -> None:
    types = ["left", "interval", "right"]
    data = summary[summary["项目"].isin(types)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [0.9, 1.6]})
    ax = axes[0]
    bars = ax.bar(data["项目"], data["数量"], color=["#0072B2", "#009E73", "#D55E00"])
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{int(row['数量'])}\n({row['比例']:.1%})", ha="center", va="bottom")
    nonmono = int(summary.loc[summary["项目"] == "nonmonotonic", "数量"].iloc[0])
    ax.text(0.98, 0.96, f"非单调轨迹：{nonmono} subjects\n该标志与三类删失不互斥", transform=ax.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set_xlabel("阈值观测类型")
    ax.set_ylabel("孕妇人数")
    ax.set_title("Y≥4% 阈值观测删失类型分布（n = subjects）")
    ax = axes[1]
    if by_bmi.empty:
        ax.text(0.5, 0.5, "没有可用的 baseline BMI 分组", ha="center", va="center")
        ax.set_axis_off()
    else:
        pivot = by_bmi.pivot(index="BMI组", columns="删失类型", values="孕妇比例").fillna(0)
        pivot = pivot.reindex(columns=types, fill_value=0)
        bottom = np.zeros(len(pivot))
        for censoring_type, color in zip(types, ["#0072B2", "#009E73", "#D55E00"]):
            values = pivot[censoring_type].to_numpy()
            ax.bar(pivot.index, values, bottom=bottom, color=color, label=censoring_type)
            bottom += values
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.set_xlabel("baseline BMI 分组（孕妇级）")
        ax.set_ylabel("组内孕妇比例")
        ax.set_title("baseline BMI 分组的阈值删失类型（n = subjects）")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(frameon=False, title="类型")
    sns.despine()
    save_figure(fig, "14_male_threshold_censoring_types.png")


def make_same_draw_repeat_groups(male: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    pair_rows = []
    eligible = male.dropna(subset=["孕妇代码", "检测抽血次数"])
    for (subject_id, draw_number), group in eligible.groupby(["孕妇代码", "检测抽血次数"], sort=True):
        if len(group) <= 1:
            continue
        ordered = group.sort_values("序号", kind="mergesort")
        y = as_numeric(ordered, Y_CONCENTRATION)
        weeks = as_numeric(ordered, WEEK)
        valid_y = y.dropna()
        mean_y = float(valid_y.mean()) if not valid_y.empty else np.nan
        sd_y = float(valid_y.std(ddof=1)) if len(valid_y) >= 2 else np.nan
        range_y = float(valid_y.max() - valid_y.min()) if not valid_y.empty else np.nan
        group_rows.append(
            {
                "孕妇代码": subject_id,
                "检测抽血次数": draw_number,
                "重复检测数": int(len(group)),
                "孕周最小值": float(weeks.min()) if weeks.notna().any() else np.nan,
                "孕周最大值": float(weeks.max()) if weeks.notna().any() else np.nan,
                "孕周范围": float(weeks.max() - weeks.min()) if weeks.notna().any() else np.nan,
                "Y均值": mean_y,
                "Y标准差": sd_y,
                "Y极差": range_y,
                "Y变异系数CV": sd_y / mean_y if np.isfinite(sd_y) and mean_y > 0 else np.nan,
                "最大绝对差": range_y,
            }
        )
        pair_data = ordered[["序号", WEEK, Y_CONCENTRATION]].copy()
        pair_data[WEEK] = as_numeric(pair_data, WEEK)
        pair_data[Y_CONCENTRATION] = as_numeric(pair_data, Y_CONCENTRATION)
        pair_data = pair_data.dropna(subset=[Y_CONCENTRATION])
        records = pair_data.to_dict("records")
        for left, right in itertools.combinations(records, 2):
            y_a = float(left[Y_CONCENTRATION])
            y_b = float(right[Y_CONCENTRATION])
            y_mean = (y_a + y_b) / 2
            difference = y_b - y_a
            pair_rows.append(
                {
                    "孕妇代码": subject_id,
                    "检测抽血次数": draw_number,
                    "记录A": left["序号"],
                    "记录B": right["序号"],
                    "孕周A": left[WEEK],
                    "孕周B": right[WEEK],
                    "Y_A": y_a,
                    "Y_B": y_b,
                    "Y_mean": y_mean,
                    "Y_difference": difference,
                    "Y_absolute_difference": abs(difference),
                    "relative_difference": abs(difference) / y_mean if y_mean > 0 else np.nan,
                }
            )
    groups = pd.DataFrame(group_rows)
    pairs = pd.DataFrame(pair_rows)
    write_table(groups, "male_same_draw_repeat_groups.csv")
    write_table(pairs, "male_same_draw_repeat_pairs.csv")
    return groups, pairs


def summarize_same_draw_repeat_error(groups: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    absolute = as_numeric(pairs, "Y_absolute_difference").dropna()
    relative = as_numeric(pairs, "relative_difference").dropna()
    difference = as_numeric(pairs, "Y_difference").dropna()
    sd_difference = float(difference.std(ddof=1)) if len(difference) >= 2 else np.nan
    mean_difference = float(difference.mean()) if not difference.empty else np.nan
    row = {
        "重复检测组数": int(len(groups)),
        "重复检测对数": int(len(pairs)),
        "Y差值平均值": mean_difference,
        "Y差值标准差": sd_difference,
        "Y绝对差中位数": float(absolute.median()) if not absolute.empty else np.nan,
        "Y绝对差Q1": float(absolute.quantile(0.25)) if not absolute.empty else np.nan,
        "Y绝对差Q3": float(absolute.quantile(0.75)) if not absolute.empty else np.nan,
        "Y绝对差95百分位数": float(absolute.quantile(0.95)) if not absolute.empty else np.nan,
        "相对差中位数": float(relative.median()) if not relative.empty else np.nan,
        "相对差95百分位数": float(relative.quantile(0.95)) if not relative.empty else np.nan,
        "mean_difference": mean_difference,
        "lower_LOA": mean_difference - 1.96 * sd_difference if np.isfinite(mean_difference) and np.isfinite(sd_difference) else np.nan,
        "upper_LOA": mean_difference + 1.96 * sd_difference if np.isfinite(mean_difference) and np.isfinite(sd_difference) else np.nan,
        "差值说明": "A/B 按序号确定性排序；同一次采血没有天然先后方向，绝对差更直接。",
    }
    result = pd.DataFrame([row])
    write_table(result, "male_same_draw_repeat_error_summary.csv")
    return result


def plot_same_draw_repeat_error(male: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups, pairs = make_same_draw_repeat_groups(male)
    summary = summarize_same_draw_repeat_error(groups, pairs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    if pairs.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "没有可计算的同次采血重复对", ha="center", va="center")
            ax.set_axis_off()
    else:
        x = as_numeric(pairs, "Y_A")
        y = as_numeric(pairs, "Y_B")
        axes[0].scatter(x, y, s=22, alpha=0.6, color="#0072B2")
        limits = [float(pd.concat([x, y]).min()), float(pd.concat([x, y]).max())]
        axes[0].plot(limits, limits, color="#D55E00", linestyle="--", linewidth=1.5, label="y=x")
        axes[0].set_xlabel("Y_A")
        axes[0].set_ylabel("Y_B")
        axes[0].set_title("重复检测 A vs B")
        axes[0].legend(frameon=False)

        pair_mean = as_numeric(pairs, "Y_mean")
        difference = as_numeric(pairs, "Y_difference")
        axes[1].scatter(pair_mean, difference, s=22, alpha=0.6, color="#009E73")
        axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[1].axhline(summary.loc[0, "lower_LOA"], color="#D55E00", linestyle=":", linewidth=1)
        axes[1].axhline(summary.loc[0, "upper_LOA"], color="#D55E00", linestyle=":", linewidth=1)
        axes[1].set_xlabel("pair mean")
        axes[1].set_ylabel("Y_difference（B−A）")
        axes[1].set_title("Bland–Altman 描述图")

        absolute = as_numeric(pairs, "Y_absolute_difference").dropna()
        axes[2].hist(absolute, bins=min(25, max(8, int(np.sqrt(len(absolute))))), color="#E69F00", edgecolor="white")
        axes[2].set_xlabel("|Y_difference|")
        axes[2].set_ylabel("重复检测对数")
        axes[2].set_title("绝对差分布")
    fig.suptitle(f"同次采血重复检测误差（groups n={len(groups)}；pairs n={len(pairs)}）", y=1.02, fontsize=13)
    fig.tight_layout()
    sns.despine()
    save_figure(fig, "15_male_same_draw_repeat_error.png")
    return groups, pairs, summary


def make_within_subject_variability(male: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = pd.read_csv(PROCESSED_DIR / "subject_summary_male.csv", encoding="utf-8-sig")
    variability = subject[subject["记录数"] >= 2].copy()
    variability_table = variability[["孕妇代码", "记录数", "Y浓度标准差", "Y浓度极差", "BMI_首次记录", BASELINE_BMI]].copy()
    write_table(variability_table, "male_within_subject_variability.csv")
    pairs = make_near_week_pairs(male, max_week_diff=0.5)
    write_table(pairs, "near_week_repeat_pairs.csv")
    return variability_table, pairs


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
        rows.append(
            {
                "孕妇代码": subject_id,
                "记录数": len(data),
                BASELINE_BMI: float(as_numeric(group, BASELINE_BMI).dropna().iloc[0]) if as_numeric(group, BASELINE_BMI).notna().any() else np.nan,
                "Y浓度对孕周线性斜率": slope,
                "正斜率": int(slope > 0),
                "负斜率": int(slope < 0),
                "0附近斜率": int(abs(slope) <= SLOPE_NEAR_ZERO_THRESHOLD),
            }
        )
    result = pd.DataFrame(rows)
    write_table(result, "male_subject_slopes.csv")
    return result


def summarize_subject_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    values = as_numeric(slopes, "Y浓度对孕周线性斜率").dropna()
    n = len(values)
    rows = [
        {"指标": "可估计斜率孕妇人数", "数值": int(n), "比例": 1.0, "说明": "至少有两条有效 Y 记录且孕周不完全相同"},
        {"指标": "正斜率人数", "数值": int((values > 0).sum()), "比例": float((values > 0).mean()) if n else np.nan, "说明": "个体内简单线性斜率大于 0"},
        {"指标": "负斜率人数", "数值": int((values < 0).sum()), "比例": float((values < 0).mean()) if n else np.nan, "说明": "个体内简单线性斜率小于 0"},
        {"指标": "0附近斜率人数", "数值": int((values.abs() <= SLOPE_NEAR_ZERO_THRESHOLD).sum()), "比例": float((values.abs() <= SLOPE_NEAR_ZERO_THRESHOLD).mean()) if n else np.nan, "说明": f"斜率绝对值不超过 {SLOPE_NEAR_ZERO_THRESHOLD:g}"},
        {"指标": "斜率中位数", "数值": float(values.median()) if n else np.nan, "比例": np.nan, "说明": "Y浓度比例值 / 周"},
        {"指标": "斜率Q1", "数值": float(values.quantile(0.25)) if n else np.nan, "比例": np.nan, "说明": "Y浓度比例值 / 周"},
        {"指标": "斜率Q3", "数值": float(values.quantile(0.75)) if n else np.nan, "比例": np.nan, "说明": "Y浓度比例值 / 周"},
    ]
    result = pd.DataFrame(rows)
    write_table(result, "male_subject_slope_summary.csv")
    return result


def plot_subject_slope_distribution(slopes: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_subject_slopes(slopes)
    values = as_numeric(slopes, "Y浓度对孕周线性斜率").dropna()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if not values.empty:
        ax.hist(values, bins=min(30, max(8, int(np.sqrt(len(values))))), color="#56B4E9", edgecolor="white")
        ax.axvline(0, color="#D55E00", linestyle="--", linewidth=1.5, label="斜率=0")
        ax.axvspan(-SLOPE_NEAR_ZERO_THRESHOLD, SLOPE_NEAR_ZERO_THRESHOLD, color="#E69F00", alpha=0.22, label="0附近区间")
    ax.set_xlabel("个体内 Y—孕周简单线性斜率（Y比例值/周）")
    ax.set_ylabel("孕妇人数")
    ax.set_title(f"个体内 Y—孕周斜率分布（n={len(values)} subjects）")
    ax.legend(frameon=False)
    sns.despine()
    save_figure(fig, "13_male_subject_slope_distribution.png")
    return summary


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


def make_female_subject_abnormal_outputs(female: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    consistency_rows = []
    for subject_id, group in female.groupby("孕妇代码", sort=True):
        labels = sorted(set(group["异常类型_分类"].dropna().astype(str)))
        t13 = int(as_numeric(group, "abnormal_T13").max())
        t18 = int(as_numeric(group, "abnormal_T18").max())
        t21 = int(as_numeric(group, "abnormal_T21").max())
        any_abnormal = int(as_numeric(group, ABNORMAL).max())
        rows.append(
            {
                "孕妇代码": subject_id,
                "记录数": int(len(group)),
                "abnormal_any_subject": any_abnormal,
                "T13_subject": t13,
                "T18_subject": t18,
                "T21_subject": t21,
                "复合异常_subject": int("复合异常" in labels),
                "标签集合": "、".join(labels),
            }
        )
        inconsistent = int(len(labels) > 1)
        if inconsistent:
            if "正常" in labels and len(labels) > 1:
                inconsistency_type = "正常与异常混合"
            else:
                inconsistency_type = "异常类型之间变化"
        else:
            inconsistency_type = "无标签变化"
        consistency_rows.append(
            {
                "孕妇代码": subject_id,
                "记录数": int(len(group)),
                "标签集合": "、".join(labels),
                "标签种类数": int(len(labels)),
                "within_subject_label_inconsistent": inconsistent,
                "不一致类型": inconsistency_type,
            }
        )
    subject_summary = pd.DataFrame(rows)
    counts_spec = [
        ("任意异常", "abnormal_any_subject"),
        ("T13", "T13_subject"),
        ("T18", "T18_subject"),
        ("T21", "T21_subject"),
        ("复合异常", "复合异常_subject"),
    ]
    counts = pd.DataFrame(
        [
            {
                "异常类型": name,
                "孕妇数": int(subject_summary[column].sum()),
                "孕妇比例": float(subject_summary[column].mean()),
                "分母孕妇数": int(len(subject_summary)),
                "统计层级": "subject",
            }
            for name, column in counts_spec
        ]
    )
    consistency = pd.DataFrame(consistency_rows)
    inconsistent_n = int(consistency["within_subject_label_inconsistent"].sum())
    consistency_summary = pd.DataFrame(
        [
            {"指标": "标签不一致孕妇", "孕妇数": inconsistent_n, "孕妇比例": inconsistent_n / len(consistency), "分母孕妇数": len(consistency)},
            {"指标": "标签一致孕妇", "孕妇数": len(consistency) - inconsistent_n, "孕妇比例": (len(consistency) - inconsistent_n) / len(consistency), "分母孕妇数": len(consistency)},
        ]
    )
    write_table(subject_summary, "female_subject_abnormal_summary.csv")
    write_table(counts, "female_subject_abnormal_counts.csv")
    write_table(consistency, "female_within_subject_label_consistency.csv")
    write_table(consistency_summary, "female_within_subject_label_consistency_summary.csv")
    return subject_summary, counts, consistency, consistency_summary


def make_female_subject_z_summary(female: pd.DataFrame, subject_abnormal: pd.DataFrame) -> pd.DataFrame:
    z_columns = {
        "Z13": "13号染色体的Z值",
        "Z18": "18号染色体的Z值",
        "Z21": "21号染色体的Z值",
        "ZX": "X染色体的Z值",
    }
    rows = []
    for subject_id, group in female.groupby("孕妇代码", sort=True):
        row = {"孕妇代码": subject_id, "记录数": int(len(group))}
        for short_name, column in z_columns.items():
            values = as_numeric(group, column).dropna()
            row[f"max_abs_{short_name}"] = float(values.abs().max()) if not values.empty else np.nan
            row[f"median_{short_name}"] = float(values.median()) if not values.empty else np.nan
        rows.append(row)
    result = pd.DataFrame(rows).merge(
        subject_abnormal[["孕妇代码", "abnormal_any_subject", "T13_subject", "T18_subject", "T21_subject"]],
        on="孕妇代码",
        how="left",
        validate="one_to_one",
    )
    write_table(result, "female_subject_z_summary.csv")
    return result


def plot_female_subject_summary(subject_z: pd.DataFrame, consistency: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    labels = ["任意异常", "T13", "T18", "T21"]
    columns = ["abnormal_any_subject", "T13_subject", "T18_subject", "T21_subject"]
    counts = [int(subject_z[column].sum()) for column in columns]
    bars = axes[0, 0].bar(labels, counts, color=["#D55E00", "#0072B2", "#009E73", "#CC79A7"])
    for bar, count in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, count + 1, str(count), ha="center", va="bottom")
    axes[0, 0].set_title(f"女胎孕妇级异常统计（n={len(subject_z)} subjects）")
    axes[0, 0].set_ylabel("孕妇人数")

    consistency_counts = consistency["不一致类型"].value_counts().reindex(["无标签变化", "正常与异常混合", "异常类型之间变化"], fill_value=0)
    axes[0, 1].bar(consistency_counts.index, consistency_counts.values, color=["#009E73", "#D55E00", "#E69F00"])
    axes[0, 1].set_title("同一孕妇标签一致性审计")
    axes[0, 1].set_ylabel("孕妇人数")
    axes[0, 1].tick_params(axis="x", rotation=18)

    for ax, short_name, title in zip(axes[1], ["Z13", "Z18"], ["max |Z13|", "max |Z18|"]):
        plot_data = subject_z[["abnormal_any_subject", f"max_abs_{short_name}"]].copy()
        plot_data["分组"] = np.where(plot_data["abnormal_any_subject"] == 1, "曾异常", "未见异常")
        sns.boxplot(data=plot_data, x="分组", y=f"max_abs_{short_name}", order=["未见异常", "曾异常"], hue="分组", palette=["#009E73", "#D55E00"], legend=False, showfliers=False, ax=ax)
        sns.stripplot(data=plot_data, x="分组", y=f"max_abs_{short_name}", order=["未见异常", "曾异常"], color="#333333", alpha=0.3, size=2, jitter=0.2, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("孕妇级异常状态")
        ax.set_ylabel("绝对 Z 值")
    fig.suptitle("女胎孕妇级异常与 Z 值辅助描述（不构成最终分类器）", y=1.0, fontsize=13)
    fig.tight_layout()
    save_figure(fig, "24_female_subject_abnormal_summary.png")


def make_variable_dictionary() -> pd.DataFrame:
    rows = [
        {"变量名": "孕周_连续值", "中文定义": "检测孕周换算后的连续周数", "层级": "record", "来源": "derived", "计算方式": "周数 + 天数/7；天数必须小于7", "单位": "周", "是否可用于后续模型输入": "是", "备注": "保留原始检测孕周"},
        {"变量名": "BMI_calc", "中文定义": "按身高和体重重算的 BMI", "层级": "record", "来源": "derived", "计算方式": "体重(kg)/(身高(cm)/100)^2", "单位": "kg/m²", "是否可用于后续模型输入": "是", "备注": "异常值只标记"},
        {"变量名": "measurement_BMI", "中文定义": "每条检测记录当时的 BMI", "层级": "record", "来源": "derived", "计算方式": "等于 BMI_calc", "单位": "kg/m²", "是否可用于后续模型输入": "是", "备注": "用于检测时 BMI 图"},
        {"变量名": "baseline_BMI", "中文定义": "孕妇最早检测孕周对应的基线 BMI", "层级": "subject", "来源": "derived", "计算方式": "最早连续孕周内 BMI_calc 的中位数，并传播到该孕妇全部记录", "单位": "kg/m²", "是否可用于后续模型输入": "是", "备注": "用于 BMI 分组与主热力图"},
        {"变量名": "Y_pass", "中文定义": "Y 浓度是否达到4%阈值", "层级": "record", "来源": "derived", "计算方式": "Y染色体浓度≥0.04为1，否则为0；缺失保留缺失", "单位": "0/1/缺失", "是否可用于后续模型输入": "可作描述或结果", "备注": "女胎结构性缺失"},
        {"变量名": "GC_range_flag", "中文定义": "GC 是否不在题面经验区间内的记录标记", "层级": "record", "来源": "derived", "计算方式": "GC<0.40或GC>0.60", "单位": "0/1", "是否可用于后续模型输入": "需谨慎", "备注": "不等于测序失败"},
        {"变量名": "abnormal_any", "中文定义": "记录是否含任意 T13/T18/T21 异常", "层级": "record", "来源": "derived", "计算方式": "三个分项标志的最大值", "单位": "0/1", "是否可用于后续模型输入": "可作结果", "备注": "不使用后验健康状态"},
        {"变量名": "abnormal_T13", "中文定义": "记录是否含 T13 异常", "层级": "record", "来源": "derived", "计算方式": "从异常文本解析 T13", "单位": "0/1", "是否可用于后续模型输入": "可作结果", "备注": "复合异常保留为1"},
        {"变量名": "abnormal_T18", "中文定义": "记录是否含 T18 异常", "层级": "record", "来源": "derived", "计算方式": "从异常文本解析 T18", "单位": "0/1", "是否可用于后续模型输入": "可作结果", "备注": "复合异常保留为1"},
        {"变量名": "abnormal_T21", "中文定义": "记录是否含 T21 异常", "层级": "record", "来源": "derived", "计算方式": "从异常文本解析 T21", "单位": "0/1", "是否可用于后续模型输入": "可作结果", "备注": "复合异常保留为1"},
        {"变量名": "胎儿是否健康", "中文定义": "题目提供的后验健康状态", "层级": "record", "来源": "raw", "计算方式": "原始附件直接保留", "单位": "分类", "是否可用于后续模型输入": "否", "备注": "仅用于描述和一致性审计，避免信息泄漏"},
        {"变量名": "首次观测达标孕周", "中文定义": "首次有效观测到 Y≥4% 的孕周", "层级": "subject", "来源": "derived", "计算方式": "按序号和孕周排序后取首次达标观测", "单位": "周", "是否可用于后续模型输入": "需谨慎", "备注": "不是无删失真实跨越时间"},
        {"变量名": "threshold_lower", "中文定义": "阈值跨越观测区间下界", "层级": "subject", "来源": "derived", "计算方式": "区间删失取首次达标前最后一次未达标孕周；右删失取最后观测孕周", "单位": "周", "是否可用于后续模型输入": "需谨慎", "备注": "左删失为空"},
        {"变量名": "threshold_upper", "中文定义": "阈值跨越观测区间上界", "层级": "subject", "来源": "derived", "计算方式": "左/区间删失取首次达标孕周", "单位": "周", "是否可用于后续模型输入": "需谨慎", "备注": "右删失为空"},
        {"变量名": "censoring_type", "中文定义": "阈值观测删失类型", "层级": "subject", "来源": "derived", "计算方式": "left、interval、right 三类互斥规则", "单位": "分类", "是否可用于后续模型输入": "需谨慎", "备注": "非单调是独立标志"},
        {"变量名": "threshold_nonmonotonic", "中文定义": "跨过阈值后再次出现低于阈值观测", "层级": "subject", "来源": "derived", "计算方式": "首次达标后是否出现 Y<0.04", "单位": "0/1", "是否可用于后续模型输入": "需谨慎", "备注": "不删除该类轨迹"},
    ]
    result = pd.DataFrame(rows)
    write_table(result, "variable_dictionary.csv")
    return result


def clear_figure_outputs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def validate_figures() -> None:
    actual = sorted(path.name for path in FIGURE_DIR.glob("*.png"))
    assert actual == sorted(FINAL_FIGURE_NAMES), f"图表文件集合异常：{actual}"
    numbers = [int(name.split("_", 1)[0]) for name in actual]
    assert sorted(numbers) == list(range(1, len(FINAL_FIGURE_NAMES) + 1)), "图号不连续或重复"
    for name in actual:
        path = FIGURE_DIR / name
        assert path.stat().st_size > 10_000, f"图表文件过小：{name}"
        with Image.open(path) as image:
            assert image.width > 0 and image.height > 0, f"图表尺寸异常：{name}"
            dpi = image.info.get("dpi")
            if dpi is not None:
                assert abs(float(dpi[0]) - 300) < 5 and abs(float(dpi[1]) - 300) < 5, f"图表 DPI 异常：{name} {dpi}"
            assert np.asarray(image.convert("L"), dtype="float32").std() > 0.5, f"疑似空白图：{name}"


def validate_report_links() -> None:
    report = REPORT_FILE.read_text(encoding="utf-8")
    targets = re.findall(r"\]\((outputs/(?:figures|tables)/[^)]+)\)", report)
    assert targets, "报告没有发现可检查的内部链接"
    missing = [target for target in targets if not (ROOT / target).exists()]
    assert not missing, f"报告存在死链：{missing}"


def report_link(filename: str) -> str:
    return f"[查看图表](outputs/figures/{filename})"


def build_report(
    male: pd.DataFrame,
    female: pd.DataFrame,
    male_y_corr: pd.DataFrame,
    bootstrap: pd.DataFrame,
    first_pass: pd.DataFrame,
    first_pass_by_bmi: pd.DataFrame,
    bmi_method: str,
    female_record_counts: pd.DataFrame,
    female_zscore: pd.DataFrame,
    female_other: pd.DataFrame,
    quality_y: pd.DataFrame,
    variability: pd.DataFrame,
    near_week_pairs: pd.DataFrame,
    slopes: pd.DataFrame,
    slope_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    same_draw_summary: pd.DataFrame,
    female_subject_abnormal: pd.DataFrame,
    female_subject_counts: pd.DataFrame,
    consistency_summary: pd.DataFrame,
    subject_z_summary: pd.DataFrame,
    font_name: str,
) -> None:
    male_subject = pd.read_csv(PROCESSED_DIR / "subject_summary_male.csv", encoding="utf-8-sig")
    female_subject = pd.read_csv(PROCESSED_DIR / "subject_summary_female.csv", encoding="utf-8-sig")
    manifest = pd.read_csv(TABLE_DIR / "data_manifest.csv", encoding="utf-8-sig")
    source_hash = manifest["原始文件SHA256"].iloc[0]
    source_path = manifest["原始文件"].iloc[0]
    sheet_names = "、".join(manifest["工作表"].astype(str))
    male_y = as_numeric(male, Y_CONCENTRATION)
    male_week = as_numeric(male, WEEK)
    pass_rate = float(male_y.ge(0.04).mean())
    repeated_male = int(male_subject["是否重复测量"].sum())
    repeated_female = int(female_subject["是否重复测量"].sum())
    male_repeated_rate = repeated_male / len(male_subject)
    female_repeated_rate = repeated_female / len(female_subject)
    female_abnormal_records = int((female[ABNORMAL] == 1).sum())
    female_abnormal_record_rate = female_abnormal_records / len(female)
    female_abnormal_subject_row = female_subject_counts.loc[female_subject_counts["异常类型"] == "任意异常"].iloc[0]
    female_abnormal_subjects = int(female_abnormal_subject_row["孕妇数"])
    female_abnormal_subject_rate = float(female_abnormal_subject_row["孕妇比例"])
    inconsistency_row = consistency_summary.loc[consistency_summary["指标"] == "标签不一致孕妇"].iloc[0]
    inconsistent_subjects = int(inconsistency_row["孕妇数"])
    inconsistent_rate = float(inconsistency_row["孕妇比例"])

    def bootstrap_row(metric: str) -> pd.Series:
        return bootstrap.loc[bootstrap["指标"] == metric].iloc[0]

    def slope_value(metric: str, column: str = "数值") -> object:
        return slope_summary.loc[slope_summary["指标"] == metric, column].iloc[0]

    week_boot = bootstrap_row("Y浓度 vs 孕周")
    bmi_boot = bootstrap_row("Y浓度 vs measurement_BMI")
    partial_boot = bootstrap_row("Y浓度 vs measurement_BMI（控制孕周）")
    week_corr = male_y_corr.loc[male_y_corr["变量"] == WEEK].iloc[0]
    bmi_corr = male_y_corr.loc[male_y_corr["变量"] == MEASUREMENT_BMI].iloc[0]
    qc_top = quality_y.sort_values("record_level_Spearman_rho", key=lambda s: s.abs(), ascending=False).iloc[0]
    censor_count = {row["项目"]: int(row["数量"]) for _, row in threshold_summary.iterrows()}
    censor_rate = {row["项目"]: float(row["比例"]) for _, row in threshold_summary.iterrows()}
    same_draw = same_draw_summary.iloc[0]

    bmi_lines = []
    for _, row in first_pass_by_bmi.dropna(subset=["BMI组"]).iterrows():
        bmi_lines.append(
            f"- {row['BMI组']}：{int(row['孕妇数'])} 名孕妇，观察到达标 {int(row['观察到达标人数'])} 名（{fmt_pct(row['观察到达标比例'])}）；"
            f"首次观测达标孕周中位数 {fmt(row['首次观测达标孕周中位数'], 2)} 周，未观测达标比例 {fmt_pct(row['未观测达标比例'])}。"
        )

    top_y_lines = []
    for _, row in male_y_corr.head(8).iterrows():
        top_y_lines.append(
            f"- {row['变量名称']}：记录层面 Spearman ρ={fmt(row['record_level_Spearman_rho'])}；"
            f"普通探索性 P={fmt_p(row['record_level_Spearman_p_naive'])}，n={int(row['record_level_n'])} records。"
        )

    z_lines = []
    for _, row in female_zscore[female_zscore["比较"].isin(["T13异常", "T18异常", "T21异常"])].iterrows():
        z_lines.append(
            f"- {row['变量名称']}（{row['比较']}）：raw Z AUC={fmt(row['AUC_raw_Z'])}，"
            f"absolute Z AUC={fmt(row['AUC_abs_Z'])}；Mann–Whitney U 的记录层面探索性 P={fmt_p(row['MannWhitney_P值'])}。"
        )

    other_lines = []
    for _, row in female_other.iterrows():
        other_lines.append(
            f"- {row['变量名称']}：正常中位数 {fmt(row['阴性中位数'])}，异常中位数 {fmt(row['阳性中位数'])}；"
            f"原始方向 AUC={fmt(row['AUC_raw'])}，绝对值 AUC={fmt(row['AUC_abs'])}。"
        )

    abnormal_subject_lines = []
    for _, row in female_subject_counts.iterrows():
        if row["异常类型"] == "任意异常":
            description = "曾至少一次出现异常标记的孕妇"
        else:
            description = f"曾至少一次出现{row['异常类型']}标记的孕妇"
        abnormal_subject_lines.append(f"{description} {int(row['孕妇数'])} 名（{fmt_pct(row['孕妇比例'])}）")

    report = f"""# 2025 C题 NIPT 数据预处理与探索性分析

## 1 数据来源与可复现性

本报告由 src/preprocess.py 和 src/eda.py 从原始 Excel 重新生成。原始文件为 {source_path}，使用工作表为 {sheet_names}；原始文件 SHA-256 为 {source_hash}。仓库中的 data/raw/附件.xlsx 只作镜像，未作为独立分析来源。

运行环境记录见 [runtime_environment.txt](outputs/tables/runtime_environment.txt)，原始来源和哈希记录见 [data_manifest.csv](outputs/tables/data_manifest.csv)。本报告只覆盖数据预处理、质量审计、描述性统计、重复测量和 EDA，不包含问题一至问题四的最终模型。

## 2 数据规模

这里同时报告记录层面和孕妇层面。记录不是相互独立的孕妇，后续统计应识别同一孕妇内的重复记录。

| 数据集 | 记录数 | 孕妇数 | 重复测量孕妇数 | 重复测量比例 |
|---|---:|---:|---:|---:|
| 男胎 | {len(male)} | {male['孕妇代码'].nunique()} | {repeated_male} | {fmt_pct(male_repeated_rate)} |
| 女胎 | {len(female)} | {female['孕妇代码'].nunique()} | {repeated_female} | {fmt_pct(female_repeated_rate)} |

男胎每名孕妇记录数为 {int(male_subject['记录数'].min())}–{int(male_subject['记录数'].max())} 条，中位数 {fmt(male_subject['记录数'].median(), 0)} 条；女胎为 {int(female_subject['记录数'].min())}–{int(female_subject['记录数'].max())} 条，中位数 {fmt(female_subject['记录数'].median(), 0)} 条。男胎约 {fmt_pct(male_repeated_rate)}、女胎约 {fmt_pct(female_repeated_rate)} 的孕妇存在重复记录，说明 record-level 与 subject-level 必须分开解释。

## 3 数据质量审计

### 3.1 原始字段、缺失与结构性缺失

- 只清理列名首尾空格，原始字段和值均保留；处理后另行追加连续孕周、BMI、阈值和重复测量分析字段。
- 孕周按“周数 + 天数 / 7”换算，例如 11w+6 为 11.857142... 周；男胎解析失败 {int(male['孕周解析失败'].sum())} 条，女胎解析失败 {int(female['孕周解析失败'].sum())} 条。失败数量保留在质量表中，没有静默转成数字。
- 处理后男胎表缺失单元格 {int(male.isna().sum().sum())} 个，女胎表缺失单元格 {int(female.isna().sum().sum())} 个。女胎 Y 染色体 Z 值和 Y 染色体浓度是结构性缺失，不填 0，也不做均值填补。
- BMI_calc 按身高和体重重算；男胎 BMI 差值绝对值中位数为 {fmt(pd.to_numeric(male['BMI_diff'], errors='coerce').abs().median(), 6)}，女胎为 {fmt(pd.to_numeric(female['BMI_diff'], errors='coerce').abs().median(), 6)}。所有有效 BMI_calc 均通过合理范围检查。
- GC_range_flag 表示 GC 不在题面经验区间 0.40–0.60 内的记录标记：男胎 {int(male['GC_range_flag'].sum())} 条，女胎 {int(female['GC_range_flag'].sum())} 条；该标记不等于测序失败。
- 比例变量、孕周范围、IQR 极端值和 BMI 一致性均只做审计或标记，不暴力删除。明细见 [missing_values.csv](outputs/tables/missing_values.csv)、[proportion_range_check.csv](outputs/tables/proportion_range_check.csv)、[bmi_consistency_summary.csv](outputs/tables/bmi_consistency_summary.csv) 和 [outlier_flags_summary.csv](outputs/tables/outlier_flags_summary.csv)。

缺失率：[01_missing_rate.png](outputs/figures/01_missing_rate.png)；BMI：[02_bmi_distribution.png](outputs/figures/02_bmi_distribution.png)；孕周：[03_gestational_week_distribution.png](outputs/figures/03_gestational_week_distribution.png)。

## 4 数据层级结构

男胎存在 {repeated_male} 名重复测量孕妇，占孕妇人数 {fmt_pct(male_repeated_rate)}；女胎存在 {repeated_female} 名重复测量孕妇，占孕妇人数 {fmt_pct(female_repeated_rate)}。这说明绝大多数孕妇存在重复记录，record-level 记录不等于 subject-level 独立样本。

完整重复行、同孕妇同日期、同孕妇同孕周和同次采血的审计见 [duplicate_audit.csv](outputs/tables/duplicate_audit.csv)。男胎同次采血重复组数为 {int((pd.read_csv(TABLE_DIR / 'duplicate_audit.csv', encoding='utf-8-sig').query("数据集 == '男胎' and 审计项 == '同次采血重复检测'")['重复组数']).iloc[0])}，女胎对应审计也保留。

同一孕妇多次出现意味着 {len(male)} 条男胎记录不等于 {male['孕妇代码'].nunique()} 个独立孕妇观测，{len(female)} 条女胎记录也不等于 {female['孕妇代码'].nunique()} 个独立孕妇观测。后续训练和验证应按孕妇分组；本报告中的普通 P 值只作为记录层面探索参考。

## 5 男胎 EDA

### 5.1 Y 浓度与 4% 阈值

男胎 Y 浓度观测范围为 {fmt(male_y.min(), 4)}–{fmt(male_y.max(), 4)}。按 Y≥0.04 定义，{len(male)} 条记录中 {int(male_y.ge(0.04).sum())} 条达标（{fmt_pct(pass_rate)}），{int(male_y.lt(0.04).sum())} 条未达标。该比例是记录层面比例，不是孕妇层面比例。

[04_y_distribution_threshold.png](outputs/figures/04_y_distribution_threshold.png)

### 5.2 Y 浓度与连续孕周

记录层面 Spearman ρ={fmt(week_corr['record_level_Spearman_rho'])}，普通 P={fmt_p(week_corr['record_level_Spearman_p_naive'])}，n={int(week_corr['record_level_n'])} records。该普通 P 把每条记录视作行级观测，不能当作独立孕妇显著性检验依据。

按孕妇整簇重采样的 cluster bootstrap 结果为：点估计 {fmt(week_boot['record_level_point_estimate'])}，中位数 {fmt(week_boot['cluster_bootstrap_median'])}，95% CI [{fmt(week_boot['CI95_lower'])}, {fmt(week_boot['CI95_upper'])}]，有效迭代 {int(week_boot['bootstrap_valid_iterations'])}/{int(week_boot['bootstrap_iterations'])}。个体内简单斜率的正斜率比例为 {fmt_pct(slope_value('正斜率人数', '比例'))}，可估计斜率孕妇数为 {int(slope_value('可估计斜率孕妇人数'))}。

因此，当前样本只能表述为记录层面与孕周存在正向单调关联，cluster bootstrap 给出了受试者内相关下的不确定性；总体相关系数和个体内纵向方向不是同一个统计量。[05_y_vs_week.png](outputs/figures/05_y_vs_week.png)

### 5.3 Y 浓度与 BMI

Y 与检测时 BMI（measurement-time BMI）的记录层面 Spearman ρ={fmt(bmi_corr['record_level_Spearman_rho'])}，普通 P={fmt_p(bmi_corr['record_level_Spearman_p_naive'])}，n={int(bmi_corr['record_level_n'])} records。按孕妇整簇重采样后，95% CI 为 [{fmt(bmi_boot['CI95_lower'])}, {fmt(bmi_boot['CI95_upper'])}]。

控制连续孕周后的部分 Spearman 点估计为 {fmt(partial_boot['record_level_point_estimate'])}，cluster bootstrap 95% CI 为 [{fmt(partial_boot['CI95_lower'])}, {fmt(partial_boot['CI95_upper'])}]。这些结果是描述性关联，不能解释为 BMI 导致 Y 浓度变化。[06_y_vs_measurement_bmi.png](outputs/figures/06_y_vs_measurement_bmi.png)

### 5.4 孕周 × baseline BMI

BMI 分组严格按孕妇级 baseline BMI：先取每名孕妇最早连续孕周的 BMI_calc；若最早孕周有多条记录则取其中位数，再传播到该孕妇的所有记录。当前采用“{bmi_method}”，每个组的孕妇人数和记录数见 [bmi_group_definitions.csv](outputs/tables/bmi_group_definitions.csv)。分组只用于 EDA，不代表问题二最终最优 BMI 分组。

[07_week_baseline_bmi_pass_heatmap.png](outputs/figures/07_week_baseline_bmi_pass_heatmap.png)

热力图的 BMI 轴使用孕妇级 baseline BMI 分组，格内达标率仍是记录层面观测比例，并同时保留记录数和孕妇数；记录数小于5的格子不显示。[08_pass_rate_by_baseline_bmi.png](outputs/figures/08_pass_rate_by_baseline_bmi.png)

BMI 组与首次观测达标的描述如下：

{chr(10).join(bmi_lines)}

### 5.5 重复测量与个体内方向

在至少两条记录的男胎孕妇中，个体内 Y 标准差中位数为 {fmt(variability['Y浓度标准差'].median(), 4)}，个体内极差中位数为 {fmt(variability['Y浓度极差'].median(), 4)}。按每名孕妇内部 Y—孕周简单线性斜率统计：正斜率 {int(slope_value('正斜率人数'))} 人（{fmt_pct(slope_value('正斜率人数', '比例'))}），负斜率 {int(slope_value('负斜率人数'))} 人（{fmt_pct(slope_value('负斜率人数', '比例'))}）；其中 {int(slope_value('0附近斜率人数'))} 人（{fmt_pct(slope_value('0附近斜率人数', '比例'))}）的绝对斜率不超过 {SLOPE_NEAR_ZERO_THRESHOLD:g}。0附近是与正、负斜率重叠的辅助标志，不是第三个互斥类别。斜率只是个体内描述，不是最终线性模型。

[09_repeated_measurement_trajectory.png](outputs/figures/09_repeated_measurement_trajectory.png) 展示记录次数较多孕妇的轨迹；[13_male_subject_slope_distribution.png](outputs/figures/13_male_subject_slope_distribution.png) 展示斜率分布。近似同孕周重复对另存于 [near_week_repeat_pairs.csv](outputs/tables/near_week_repeat_pairs.csv)，不与同次采血重复检测混用。

### 5.6 阈值观测与删失

“首次观测达标孕周”只表示第一次观测到 Y≥4%，不是无删失的真实跨阈值时刻。每名男胎孕妇恰好一行的阈值表见 [male_threshold_censoring.csv](outputs/tables/male_threshold_censoring.csv)，类型统计见 [threshold_censoring_summary.csv](outputs/tables/threshold_censoring_summary.csv)。

- left：{censor_count.get('left', 0)} 名（{censor_rate.get('left', 0):.1%}），只能知道真实跨越时间不晚于首次观测。
- interval：{censor_count.get('interval', 0)} 名（{censor_rate.get('interval', 0):.1%}），跨越时间位于最后一次未达标和首次达标之间。
- right：{censor_count.get('right', 0)} 名（{censor_rate.get('right', 0):.1%}），截至最后观测仍未观察到达标。
- 非单调轨迹：{censor_count.get('nonmonotonic', 0)} 名（{censor_rate.get('nonmonotonic', 0):.1%}）；这是独立标志，与三类删失不互斥。

按 baseline BMI 分组的删失类型统计见 [threshold_censoring_by_bmi.csv](outputs/tables/threshold_censoring_by_bmi.csv)，图 14 同时展示总体类型和组内类型比例。

[10_first_observed_pass_week_distribution.png](outputs/figures/10_first_observed_pass_week_distribution.png)；[11_first_observed_pass_week_vs_baseline_bmi.png](outputs/figures/11_first_observed_pass_week_vs_baseline_bmi.png)；[14_male_threshold_censoring_types.png](outputs/figures/14_male_threshold_censoring_types.png)。

### 5.7 同次采血重复检测误差

男胎同次采血重复检测单独按孕妇代码和检测抽血次数分组，共 {int(same_draw['重复检测组数'])} 个重复组、{int(same_draw['重复检测对数'])} 个重复对。Y 绝对差中位数为 {fmt(same_draw['Y绝对差中位数'], 6)}，95 百分位数为 {fmt(same_draw['Y绝对差95百分位数'], 6)}；相对差中位数为 {fmt_pct(same_draw['相对差中位数'])}，95 百分位数为 {fmt_pct(same_draw['相对差95百分位数'])}。差值和 Bland–Altman 描述性界限见 [male_same_draw_repeat_error_summary.csv](outputs/tables/male_same_draw_repeat_error_summary.csv)。

A/B 按序号确定性排序；同一次采血没有天然先后测量方向，差值符号只作描述，绝对差更加直接。不要把这部分与近似同孕周重复对混为一谈。[15_male_same_draw_repeat_error.png](outputs/figures/15_male_same_draw_repeat_error.png)

### 5.8 测序质量与 Y

与 Y 浓度绝对值关联最大的质量变量是 {qc_top['变量名称']}，记录层面 Spearman ρ={fmt(qc_top['record_level_Spearman_rho'])}，普通探索性 P={fmt_p(qc_top['record_level_Spearman_p_naive'])}，n={int(qc_top['record_level_n'])} records。完整表见 [quality_vs_y_summary.csv](outputs/tables/quality_vs_y_summary.csv)，图见 [16_quality_vs_y.png](outputs/figures/16_quality_vs_y.png)。

这些质量变量与 Y 存在统计关联，因此后续可以把它们作为质量协变量或敏感性分析变量；当前 EDA 不据此作因果判断。

候选变量的记录层面关联排序如下：

{chr(10).join(top_y_lines)}

## 6 女胎 EDA

### 6.1 记录级异常分布

女胎共有 {len(female)} 条 records，其中任意异常 {female_abnormal_records} 条（{fmt_pct(female_abnormal_record_rate)}），正常 {len(female) - female_abnormal_records} 条。该比例是异常记录比例。[17_female_abnormal_distribution.png](outputs/figures/17_female_abnormal_distribution.png)

记录级类型统计见 [female_abnormal_counts.csv](outputs/tables/female_abnormal_counts.csv)：{'；'.join([f"{row['异常类型']} {int(row['数量'])} 条" for _, row in female_record_counts.iterrows()])}。

### 6.2 孕妇级异常分布

{len(female_subject_abnormal)} 名孕妇中，孕妇级统计为：{'；'.join(abnormal_subject_lines)}。孕妇级结果见 [female_subject_abnormal_summary.csv](outputs/tables/female_subject_abnormal_summary.csv) 和 [female_subject_abnormal_counts.csv](outputs/tables/female_subject_abnormal_counts.csv)。

记录级异常数与孕妇级异常数不能互相替代；同一孕妇只要任一检测记录出现对应标志，孕妇级标志就记为1。记录级任意异常标记为 {female_abnormal_records} 条；孕妇级则是 {female_abnormal_subjects} 名孕妇曾至少一次出现异常标记（{fmt_pct(female_abnormal_subject_rate)}），不应直接解释为最终真实异常患病率。

### 6.3 同一孕妇标签一致性

标签一致性按每名孕妇的异常类型集合审计，不决定哪个检测标签是真值。共 {inconsistent_subjects} 名孕妇（{fmt_pct(inconsistent_rate)}）存在不同记录标签，细节见 [female_within_subject_label_consistency.csv](outputs/tables/female_within_subject_label_consistency.csv)，汇总见 [female_within_subject_label_consistency_summary.csv](outputs/tables/female_within_subject_label_consistency_summary.csv)。

### 6.4 Z13、Z18、Z21 的两种合法描述

对于每个对应异常，均同时给出原始方向 Z 的 AUC 和绝对值 Z 的 AUC；二者不择一。P 值是记录层面探索性 Mann–Whitney U P 值，存在重复测量，不能解释成独立孕妇检验。

{chr(10).join(z_lines)}

[18_z13_vs_t13.png](outputs/figures/18_z13_vs_t13.png)　[19_z18_vs_t18.png](outputs/figures/19_z18_vs_t18.png)　[20_z21_vs_t21.png](outputs/figures/20_z21_vs_t21.png)

X 染色体 Z 值与任意异常的辅助图见 [21_zx_vs_abnormal.png](outputs/figures/21_zx_vs_abnormal.png)。完整新版结果见 [female_zscore_discrimination_summary.csv](outputs/tables/female_zscore_discrimination_summary.csv)。

### 6.5 孕妇级 Z 值辅助描述与其他特征

每名女胎孕妇的 max_abs_Z13、max_abs_Z18、max_abs_Z21、max_abs_ZX 和对应中位数见 [female_subject_z_summary.csv](outputs/tables/female_subject_z_summary.csv)。这只是孕妇级描述性聚合，不是最终分类器表现。

其他候选变量的记录层面描述同时保留原始方向 AUC 和绝对值 AUC，不使用单一指标作最终特征筛选：

{chr(10).join(other_lines)}

[23_female_other_features_vs_abnormal.png](outputs/figures/23_female_other_features_vs_abnormal.png)；女胎候选变量热力图见 [22_female_spearman_heatmap.png](outputs/figures/22_female_spearman_heatmap.png)；孕妇级异常、一致性和 Z 值辅助图见 [24_female_subject_abnormal_summary.png](outputs/figures/24_female_subject_abnormal_summary.png)。

## 7 最可靠的 EDA 结论

1. 记录层面，男胎 Y 浓度与孕周、检测时 BMI 均呈现关联；普通 P 值仅作行级探索参考，主要不确定性应参考按孕妇聚类的 bootstrap 95% CI。
2. 个体内简单斜率方向与总体记录层面相关系数回答不同问题，当前样本不能把二者合并成一个“每名孕妇都同速变化”的结论。
3. baseline BMI 与 measurement-time BMI 定义不同；BMI 分组和主热力图使用前者，检测时散点图使用后者。
4. 首次观测达标包含 left、interval、right 三类观测删失，非单调轨迹另作标志；首次观测周不能直接当作真实阈值跨越时间。
5. 同次采血重复检测提供了更直接的检测重复性描述，不能与近似同孕周重复对混合。
6. 女胎必须同时看记录级和孕妇级异常统计，且同一孕妇的不同标签需单独审计；Z 值同时报告原始方向和绝对值两种 AUC。
7. 测序质量变量与 Y 的关联只能支持质量协变量或敏感性分析建议，不能据此作因果判断。

### 推荐进入论文的核心图

下列图标记为“推荐进入论文”，其余图用于附录或内部审计：

1. 推荐进入论文：Y 浓度分布与4%阈值——[04_y_distribution_threshold.png](outputs/figures/04_y_distribution_threshold.png)
2. 推荐进入论文：Y 与孕周及 LOWESS——[05_y_vs_week.png](outputs/figures/05_y_vs_week.png)
3. 推荐进入论文：Y 与检测时 BMI 及 LOWESS——[06_y_vs_measurement_bmi.png](outputs/figures/06_y_vs_measurement_bmi.png)
4. 推荐进入论文：孕周 × baseline BMI 达标率热力图——[07_week_baseline_bmi_pass_heatmap.png](outputs/figures/07_week_baseline_bmi_pass_heatmap.png)
5. 推荐进入论文：baseline BMI 分组达标率—孕周——[08_pass_rate_by_baseline_bmi.png](outputs/figures/08_pass_rate_by_baseline_bmi.png)
6. 推荐进入论文：男胎纵向轨迹——[09_repeated_measurement_trajectory.png](outputs/figures/09_repeated_measurement_trajectory.png)
7. 推荐进入论文：阈值删失类型与首次观测达标——[10_first_observed_pass_week_distribution.png](outputs/figures/10_first_observed_pass_week_distribution.png) 和 [14_male_threshold_censoring_types.png](outputs/figures/14_male_threshold_censoring_types.png)
8. 推荐进入论文：同次采血重复检测误差——[15_male_same_draw_repeat_error.png](outputs/figures/15_male_same_draw_repeat_error.png)
9. 推荐进入论文：男胎相关热力图——[12_male_spearman_heatmap.png](outputs/figures/12_male_spearman_heatmap.png)
10. 推荐进入论文：女胎异常类型——[17_female_abnormal_distribution.png](outputs/figures/17_female_abnormal_distribution.png)
11. 推荐进入论文：Z13、Z18、Z21 异常比较——[18_z13_vs_t13.png](outputs/figures/18_z13_vs_t13.png)、[19_z18_vs_t18.png](outputs/figures/19_z18_vs_t18.png)、[20_z21_vs_t21.png](outputs/figures/20_z21_vs_t21.png)
12. 推荐进入论文：女胎孕妇级异常与标签一致性——[24_female_subject_abnormal_summary.png](outputs/figures/24_female_subject_abnormal_summary.png)

## 8 给后续建模人员的数据提示

- 数据存在 repeated measures，训练、验证和交叉验证应按孕妇分组。
- threshold time 存在 observation censoring，首次观测周不能直接作为真实跨越时刻。
- baseline BMI 与 measurement BMI 不同，使用时应保留定义来源。
- same-draw 数据可用于描述检测误差，不能和近似同孕周重复混用。
- 女胎标签存在重复记录和 within-subject label inconsistency，应先审计再聚合。
- 后验健康状态只能用于描述和一致性检查，禁止作为异常预测输入造成信息泄漏。

### 8.1 可复现文件与范围边界

主要脚本：src/preprocess.py、src/eda.py。

处理数据：data/processed/male_cleaned.csv、data/processed/female_cleaned.csv、data/processed/subject_summary_male.csv、data/processed/subject_summary_female.csv。

重要表格：cluster_bootstrap_correlations.csv、male_subject_slope_summary.csv、bmi_group_definitions.csv、male_threshold_censoring.csv、threshold_censoring_summary.csv、threshold_censoring_by_bmi.csv、male_same_draw_repeat_groups.csv、male_same_draw_repeat_pairs.csv、male_same_draw_repeat_error_summary.csv、female_zscore_discrimination_summary.csv、female_subject_abnormal_summary.csv、female_subject_abnormal_counts.csv、female_within_subject_label_consistency.csv、female_subject_z_summary.csv、variable_dictionary.csv、runtime_environment.txt。

辅助表格：male_first_pass_week.csv、first_pass_week_by_bmi.csv、male_within_subject_variability.csv、near_week_repeat_pairs.csv、correlation_y.csv、quality_vs_y_summary.csv、health_abnormal_consistency.csv。

图表为统一中文风格，PNG 由脚本以 300 dpi 保存；本次字体回退为 {font_name}。完整数字来自脚本运行时的表格，不手工固定样本量。

> 本报告严格限定于原始数据审计、预处理、数据质量检查、特征整理、重复测量描述、检测误差描述、描述性统计、可视化和 EDA；未修改问题一最终模型，也未完成问题二、问题三或问题四最终模型。
"""
    REPORT_FILE.write_text(report.rstrip() + "\n", encoding="utf-8")

def assert_eda_inputs(male: pd.DataFrame, female: pd.DataFrame) -> None:
    assert len(male) == 1082 and male["孕妇代码"].nunique() == 267
    assert len(female) == 605 and female["孕妇代码"].nunique() == 147
    assert male[PASS].dropna().isin([0, 1]).all()
    assert female[Y_CONCENTRATION].isna().all() and female["Y染色体Z值"].isna().all()
    for column in ["胎儿是否健康", "异常类型_分类"]:
        assert column not in MALE_CORRELATION + FEMALE_CORRELATION
    assert male.groupby("孕妇代码")[BASELINE_BMI].nunique(dropna=True).le(1).all()
    assert female.groupby("孕妇代码")[ABNORMAL].max().isin([0, 1]).all()


def main() -> None:
    np.random.seed(BOOTSTRAP_SEED)
    font_name = configure_plot_style()
    male = load_processed("male_cleaned.csv")
    female = load_processed("female_cleaned.csv")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    assert_eda_inputs(male, female)
    clear_figure_outputs()
    make_variable_dictionary()

    male_grouped, _, bmi_definitions, bmi_method = prepare_bmi_grouping(male)
    bootstrap = cluster_bootstrap_correlations(male)
    threshold = make_threshold_censoring(male_grouped)
    threshold_summary = summarize_threshold_censoring(threshold)
    threshold_by_bmi = summarize_threshold_censoring_by_bmi(threshold)
    plot_threshold_censoring_types(threshold_summary, threshold_by_bmi)

    plot_missing_rate(male, female)
    plot_bmi_distribution(male, female)
    plot_week_distribution(male, female)
    plot_y_distribution(male)
    week_boot = bootstrap.loc[bootstrap["指标"] == "Y浓度 vs 孕周"].iloc[0]
    bmi_boot = bootstrap.loc[bootstrap["指标"] == "Y浓度 vs measurement_BMI"].iloc[0]
    plot_y_vs_week(male, week_boot)
    plot_y_vs_bmi(male, bmi_boot)
    plot_week_bmi_heatmap(male_grouped, bmi_definitions)
    plot_pass_rate_by_bmi(male_grouped, bmi_definitions, bmi_method)
    plot_trajectories(male_grouped)
    first_pass, first_pass_by_bmi = plot_first_pass(threshold, bmi_definitions, bmi_method)
    male_y_corr = plot_male_correlation(male)

    female_record_counts = plot_female_abnormal_distribution(female)
    discrimination_rows = []
    target_specs = [
        ("13号染色体的Z值", "abnormal_T13", "T13异常", "18_z13_vs_t13.png"),
        ("18号染色体的Z值", "abnormal_T18", "T18异常", "19_z18_vs_t18.png"),
        ("21号染色体的Z值", "abnormal_T21", "T21异常", "20_z21_vs_t21.png"),
    ]
    for feature, target, title, filename in target_specs:
        comparison = plot_z_comparison(female, feature, target, filename, f"{label(feature)}与{title}")
        comparison["比较"] = title
        discrimination_rows.append(comparison)
    discrimination_rows.append(plot_zx_comparison(female))
    female_zscore = pd.DataFrame(discrimination_rows)
    write_table(female_zscore, "female_zscore_discrimination_summary.csv")
    legacy_zscore = female_zscore.copy()
    legacy_zscore.insert(0, "表格状态", "legacy_compatibility")
    write_table(legacy_zscore, "female_discrimination_summary.csv")

    female_other = plot_female_other_features(female)
    plot_female_correlation(female)
    quality_y = plot_quality_vs_y(male)
    variability, near_week_pairs = make_within_subject_variability(male)
    slopes = compute_subject_slopes(male)
    slope_summary = plot_subject_slope_distribution(slopes)
    same_draw_groups, same_draw_pairs, same_draw_summary = plot_same_draw_repeat_error(male)
    health_consistency = create_health_consistency_table(male, female)
    female_subject_abnormal, female_subject_counts, consistency, consistency_summary = make_female_subject_abnormal_outputs(female)
    subject_z_summary = make_female_subject_z_summary(female, female_subject_abnormal)
    plot_female_subject_summary(subject_z_summary, consistency)

    plot_univariate_panels(male, MALE_UNIVARIATE, "男胎", "25_male_univariate_panels.png")
    plot_univariate_panels(female, FEMALE_UNIVARIATE, "女胎", "26_female_univariate_panels.png")
    validate_figures()

    build_report(
        male=male,
        female=female,
        male_y_corr=male_y_corr,
        bootstrap=bootstrap,
        first_pass=first_pass,
        first_pass_by_bmi=first_pass_by_bmi,
        bmi_method=bmi_method,
        female_record_counts=female_record_counts,
        female_zscore=female_zscore,
        female_other=female_other,
        quality_y=quality_y,
        variability=variability,
        near_week_pairs=near_week_pairs,
        slopes=slopes,
        slope_summary=slope_summary,
        threshold_summary=threshold_summary,
        same_draw_summary=same_draw_summary,
        female_subject_abnormal=female_subject_abnormal,
        female_subject_counts=female_subject_counts,
        consistency_summary=consistency_summary,
        subject_z_summary=subject_z_summary,
        font_name=font_name,
    )
    validate_report_links()
    print(f"EDA 完成：生成 {len(list(FIGURE_DIR.glob('*.png')))} 张 PNG 图和报告 {REPORT_FILE.name}。")


if __name__ == "__main__":
    main()
