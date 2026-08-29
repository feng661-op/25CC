"""2025 年 C 题 NIPT 原始附件预处理与数据质量审计。

脚本只追加分析字段，不删除原始记录或异常值。请从仓库根目录运行：
    python src/preprocess.py
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_RAW_FILE = ROOT / "2025C原题" / "附件.xlsx"
MIRROR_RAW_FILE = ROOT / "data" / "raw" / "附件.xlsx"
RAW_FILE = REPOSITORY_RAW_FILE if REPOSITORY_RAW_FILE.exists() else MIRROR_RAW_FILE
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"

MALE_SHEET_HINT = "男胎"
FEMALE_SHEET_HINT = "女胎"

RATIO_COLUMNS = [
    "在参考基因组上比对的比例",
    "重复读段的比例",
    "被过滤掉读段数的比例",
]
GC_COLUMN = "GC含量"
Y_CONCENTRATION_COLUMN = "Y染色体浓度"
Y_Z_COLUMN = "Y染色体Z值"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """只清理列名首尾空格，保留原始字段和值。"""
    columns = [str(column).strip() for column in df.columns]
    if len(columns) != len(set(columns)):
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        raise ValueError(f"列名去除首尾空格后出现重复：{duplicates}")
    df = df.copy()
    df.columns = columns
    return df


def parse_gestational_week(value: object) -> float:
    """将 11w+6 等孕周转换为 11 + 6/7，不把 6 天写成 0.6 周。"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip().lower().replace("＋", "+").replace("周", "w")
    text = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)w(?:\+(\d+))?", text)
    if match:
        weeks = float(match.group(1))
        days = int(match.group(2) or 0)
        if days >= 7:
            return np.nan
        return weeks + days / 7.0

    # 兼容已经是连续周数的数值字符串；原始题目格式仍优先按上面的规则解析。
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    return np.nan


def parse_date_value(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value)
    text = str(value).strip()
    if re.fullmatch(r"\d{8}(?:\.0)?", text):
        return pd.to_datetime(text[:8], format="%Y%m%d", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def iqr_flag(series: pd.Series) -> tuple[pd.Series, float, float]:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index, dtype="boolean"), np.nan, np.nan
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flags = ((series < lower) | (series > upper)).fillna(False).astype("boolean")
    return flags, float(lower), float(upper)


def abnormal_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return sorted(set(re.findall(r"T(?:13|18|21)", str(value).upper())))


def add_subject_fields(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    """添加孕妇层级重复测量字段；每条原始记录仍保留。"""
    df = df.copy()
    subject = df["孕妇代码"].astype("string")

    record_count = subject.value_counts(dropna=False)
    df["孕妇记录数"] = subject.map(record_count).astype("Int64")

    df["孕妇不同孕周数"] = (
        df.groupby(subject, dropna=False)["孕周_连续值"].transform("nunique").astype("Int64")
    )
    df["孕妇检测日期数"] = (
        df.groupby(subject, dropna=False)["检测日期_日期"].transform("nunique").astype("Int64")
    )
    df["孕妇检测抽血次数数"] = (
        df.groupby(subject, dropna=False)["检测抽血次数"].transform("nunique").astype("Int64")
    )

    draw_key = df["检测抽血次数"]
    df["同次采血检测次数"] = (
        df.groupby([subject, draw_key], dropna=False)["序号"].transform("size").astype("Int64")
    )
    week_key = df["孕周_连续值"].round(8)
    df["同孕周检测次数"] = (
        df.groupby([subject, week_key], dropna=False)["序号"].transform("size").astype("Int64")
    )
    df["是否重复测量"] = (df["孕妇记录数"] > 1).astype("boolean")
    df["同次采血重复测量"] = (df["同次采血检测次数"] > 1).astype("boolean")

    if sex == "男胎":
        y = numeric(df, Y_CONCENTRATION_COLUMN)
        y_pass = pd.Series(pd.NA, index=df.index, dtype="Int64")
        valid = y.notna()
        y_pass.loc[valid] = (y.loc[valid] >= 0.04).astype("int64")
        df["Y_pass"] = y_pass
    else:
        df["Y_pass"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    return df


def add_analysis_fields(raw_df: pd.DataFrame, sex: str) -> tuple[pd.DataFrame, list[str], dict[str, tuple[float, float]]]:
    """保留原始字段并追加标准化字段。"""
    df = clean_column_names(raw_df)
    source_columns = list(df.columns)

    # 女胎原始表中相应列没有标题且全为空；补充统一分析字段，但不覆盖原始空白列。
    if sex == "女胎":
        if Y_Z_COLUMN not in df.columns:
            df[Y_Z_COLUMN] = np.nan
        if Y_CONCENTRATION_COLUMN not in df.columns:
            df[Y_CONCENTRATION_COLUMN] = np.nan
    elif Y_CONCENTRATION_COLUMN not in df.columns:
        raise ValueError("男胎工作表缺少 Y 染色体浓度字段")

    week = df["检测孕周"].map(parse_gestational_week)
    date_column = df["检测日期"].map(parse_date_value)
    lmp_column = df["末次月经"].map(parse_date_value)
    df["检测日期_日期"] = date_column
    df["末次月经_日期"] = lmp_column
    df["孕周_连续值"] = week.astype("float64")
    df["孕周解析失败"] = (df["检测孕周"].notna() & week.isna()).astype("boolean")

    height_cm = numeric(df, "身高")
    weight_kg = numeric(df, "体重")
    bmi_original = numeric(df, "孕妇BMI")
    bmi_calc = weight_kg / (height_cm / 100.0) ** 2
    bmi_calc = bmi_calc.where((height_cm > 0) & (weight_kg > 0))
    df["BMI_calc"] = bmi_calc
    df["BMI_diff"] = bmi_original - bmi_calc

    y_z = numeric(df, Y_Z_COLUMN)
    y_concentration = numeric(df, Y_CONCENTRATION_COLUMN)
    df["Y相关字段_结构性缺失"] = (y_z.isna() & y_concentration.isna()).astype("boolean")

    abnormal_text = df["染色体的非整倍体"].fillna("").astype("string").str.strip()
    tokens = abnormal_text.map(abnormal_tokens)
    df["abnormal_any"] = tokens.map(lambda values: int(bool(values))).astype("Int64")
    df["abnormal_T13"] = tokens.map(lambda values: int("T13" in values)).astype("Int64")
    df["abnormal_T18"] = tokens.map(lambda values: int("T18" in values)).astype("Int64")
    df["abnormal_T21"] = tokens.map(lambda values: int("T21" in values)).astype("Int64")
    df["异常类型_分类"] = tokens.map(
        lambda values: "正常" if not values else (values[0] if len(values) == 1 else "复合异常")
    )

    df = add_subject_fields(df, sex)

    df["GC_abnormal"] = (
        numeric(df, GC_COLUMN).notna() & ~numeric(df, GC_COLUMN).between(0.40, 0.60)
    ).astype("boolean")
    df["孕周异常"] = (
        (week.notna() & ~week.between(8.0, 42.0)) | (df["孕周解析失败"] == True)
    ).astype("boolean")

    ratio_flags = []
    for column in RATIO_COLUMNS:
        values = numeric(df, column)
        ratio_flags.append(values.notna() & ~values.between(0.0, 1.0))
    if ratio_flags:
        df["比例变量超界"] = pd.concat(ratio_flags, axis=1).any(axis=1).astype("boolean")
    else:
        df["比例变量超界"] = pd.Series(False, index=df.index, dtype="boolean")

    outlier_bounds: dict[str, tuple[float, float]] = {}
    for flag_name, column in [
        ("outlier_BMI", "BMI_calc"),
        ("outlier_GC", GC_COLUMN),
        ("outlier_Y", Y_CONCENTRATION_COLUMN),
        ("outlier_reads", "原始读段数"),
    ]:
        flags, lower, upper = iqr_flag(numeric(df, column))
        df[flag_name] = flags
        outlier_bounds[flag_name] = (lower, upper)

    return df, source_columns, outlier_bounds


def first_valid(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


def make_subject_summary(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_id, group in df.groupby("孕妇代码", dropna=False, sort=True):
        ordered = group.sort_values(["孕周_连续值", "检测日期_日期", "序号"], na_position="last")
        week = numeric(ordered, "孕周_连续值")
        bmi = numeric(ordered, "BMI_calc")
        summary: dict[str, object] = {
            "孕妇代码": subject_id,
            "记录数": int(len(group)),
            "不同检测孕周数": int(group["孕周_连续值"].nunique(dropna=True)),
            "检测日期数": int(group["检测日期_日期"].nunique(dropna=True)),
            "检测抽血次数数": int(group["检测抽血次数"].nunique(dropna=True)),
            "最大同次采血检测次数": int(group["同次采血检测次数"].max()),
            "是否重复测量": bool(len(group) > 1),
            "最早孕周": float(week.min()) if week.notna().any() else np.nan,
            "最晚孕周": float(week.max()) if week.notna().any() else np.nan,
            "BMI_首次记录": first_valid(bmi),
            "BMI_中位数": float(bmi.median()) if bmi.notna().any() else np.nan,
            "首次检测日期": first_valid(ordered["检测日期_日期"]),
            "异常是否存在": int(group["abnormal_any"].max()) if "abnormal_any" in group else np.nan,
        }
        if sex == "男胎":
            y = numeric(group, Y_CONCENTRATION_COLUMN)
            pass_mask = y >= 0.04
            pass_weeks = week.loc[pass_mask.reindex(week.index, fill_value=False)]
            first_observed_week = summary["最早孕周"]
            first_pass_week = float(pass_weeks.min()) if pass_weeks.notna().any() else np.nan
            summary.update(
                {
                    "Y浓度均值": float(y.mean()) if y.notna().any() else np.nan,
                    "Y浓度标准差": float(y.std(ddof=1)) if y.notna().sum() >= 2 else np.nan,
                    "Y浓度极差": float(y.max() - y.min()) if y.notna().any() else np.nan,
                    "达标记录数": int(pass_mask.sum()),
                    "首次达标孕周": first_pass_week,
                    "未观测达标": int(not np.isfinite(first_pass_week)),
                    "首次观测即达标": int(
                        np.isfinite(first_pass_week)
                        and np.isfinite(first_observed_week)
                        and np.isclose(first_pass_week, first_observed_week)
                    ),
                }
            )
        rows.append(summary)
    return pd.DataFrame(rows)


def repeated_group_metrics(df: pd.DataFrame, columns: list[str]) -> tuple[int, int, int]:
    valid = df.dropna(subset=columns)
    sizes = valid.groupby(columns, dropna=False).size()
    repeated = sizes[sizes > 1]
    repeated_rows = int(repeated.sum())
    repeated_groups = int(repeated.size)
    pairs = int(sum(n * (n - 1) // 2 for n in repeated.tolist()))
    return repeated_groups, repeated_rows, pairs


def make_duplicate_audit(raw_df: pd.DataFrame, df: pd.DataFrame, sex: str) -> pd.DataFrame:
    same_date = repeated_group_metrics(df, ["孕妇代码", "检测日期_日期"])
    same_week = repeated_group_metrics(df, ["孕妇代码", "孕周_连续值"])
    same_draw = repeated_group_metrics(df, ["孕妇代码", "检测抽血次数"])
    rows = [
        {
            "数据集": sex,
            "审计项": "完全重复行",
            "重复组数": int(raw_df.duplicated().sum()),
            "涉及行数": int(raw_df.duplicated(keep=False).sum()),
            "重复对数": np.nan,
            "说明": "只标记，不删除",
        },
        {
            "数据集": sex,
            "审计项": "同孕妇同检测日期",
            "重复组数": same_date[0],
            "涉及行数": same_date[1],
            "重复对数": same_date[2],
            "说明": "排除缺失检测日期",
        },
        {
            "数据集": sex,
            "审计项": "同孕妇同连续孕周",
            "重复组数": same_week[0],
            "涉及行数": same_week[1],
            "重复对数": same_week[2],
            "说明": "近似同孕周按解析后数值完全相同检查",
        },
        {
            "数据集": sex,
            "审计项": "同次采血重复检测",
            "重复组数": same_draw[0],
            "涉及行数": same_draw[1],
            "重复对数": same_draw[2],
            "说明": "同一孕妇、同一抽血次数下的多条记录",
        },
    ]
    return pd.DataFrame(rows)


def make_quality_table(df: pd.DataFrame, source_columns: list[str], sex: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "数据集": sex,
                "字段": column,
                "字段来源": "原始字段" if column in source_columns else "分析字段",
                "数据类型": str(series.dtype),
                "非空数量": int(series.notna().sum()),
                "缺失数量": missing,
                "缺失率": missing / len(df) if len(df) else np.nan,
                "唯一值数量": int(series.nunique(dropna=True)),
                "结构性缺失": int(sex == "女胎" and column in {Y_Z_COLUMN, Y_CONCENTRATION_COLUMN}),
            }
        )
    return pd.DataFrame(rows)


def make_numeric_statistics(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
            continue
        values = pd.to_numeric(df[column], errors="coerce").astype("float64").dropna()
        if values.empty:
            stats = {key: np.nan for key in ["平均值", "标准差", "最小值", "Q1", "中位数", "Q3", "最大值"]}
        else:
            quantiles = values.quantile([0.25, 0.50, 0.75])
            stats = {
                "平均值": float(values.mean()),
                "标准差": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
                "最小值": float(values.min()),
                "Q1": float(quantiles.loc[0.25]),
                "中位数": float(quantiles.loc[0.50]),
                "Q3": float(quantiles.loc[0.75]),
                "最大值": float(values.max()),
            }
        rows.append({"数据集": sex, "字段": column, "非空数量": int(values.size), **stats})
    return pd.DataFrame(rows)


def make_category_counts(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    columns = [
        "IVF妊娠",
        "检测抽血次数",
        "怀孕次数",
        "生产次数",
        "异常类型_分类",
        "染色体的非整倍体",
        "胎儿是否健康",
        "Y_pass",
        "GC_abnormal",
        "是否重复测量",
    ]
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].astype("object").where(df[column].notna(), "空白/缺失")
        counts = values.value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "数据集": sex,
                    "字段": column,
                    "类别": str(value),
                    "数量": int(count),
                    "比例": float(count / len(df)),
                }
            )
    return pd.DataFrame(rows)


def make_proportion_check(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    rows = []
    for column in RATIO_COLUMNS:
        values = numeric(df, column)
        out = values.notna() & ~values.between(0.0, 1.0)
        rows.append(
            {
                "数据集": sex,
                "字段": column,
                "非空数量": int(values.notna().sum()),
                "超出[0,1]数量": int(out.sum()),
                "超界比例": float(out.sum() / values.notna().sum()) if values.notna().any() else np.nan,
                "最小值": float(values.min()) if values.notna().any() else np.nan,
                "最大值": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_bmi_consistency(df: pd.DataFrame, sex: str) -> pd.DataFrame:
    diff = numeric(df, "BMI_diff").dropna()
    abs_diff = diff.abs()
    return pd.DataFrame(
        [
            {
                "数据集": sex,
                "可比较行数": int(diff.size),
                "BMI差值平均值": float(diff.mean()) if not diff.empty else np.nan,
                "BMI差值标准差": float(diff.std(ddof=1)) if diff.size >= 2 else np.nan,
                "BMI差值中位数": float(diff.median()) if not diff.empty else np.nan,
                "BMI差值绝对值中位数": float(abs_diff.median()) if not abs_diff.empty else np.nan,
                "绝对差值不超过0.1行数": int((abs_diff <= 0.1).sum()),
                "绝对差值不超过0.1比例": float((abs_diff <= 0.1).mean()) if not abs_diff.empty else np.nan,
                "绝对差值超过0.5行数": int((abs_diff > 0.5).sum()),
                "最大绝对差值": float(abs_diff.max()) if not abs_diff.empty else np.nan,
            }
        ]
    )


def make_outlier_summary(df: pd.DataFrame, sex: str, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    rules = {
        "outlier_BMI": "BMI_calc 按本数据集 IQR 的 1.5 倍规则，仅标记",
        "outlier_GC": "GC含量按本数据集 IQR 的 1.5 倍规则，仅标记",
        "outlier_Y": "Y染色体浓度按本数据集 IQR 的 1.5 倍规则，仅标记",
        "outlier_reads": "原始读段数按本数据集 IQR 的 1.5 倍规则，仅标记",
        "GC_abnormal": "GC含量不在 0.40 至 0.60 内，仅标记",
        "孕周异常": "孕周解析失败或不在 8 至 42 周内，仅标记",
        "比例变量超界": "比例变量不在 0 至 1 内，仅标记",
    }
    for flag, rule in rules.items():
        values = df[flag].fillna(False).astype(bool)
        lower, upper = bounds.get(flag, (np.nan, np.nan))
        rows.append(
            {
                "数据集": sex,
                "标志": flag,
                "标记行数": int(values.sum()),
                "标记比例": float(values.mean()) if len(values) else np.nan,
                "下界": lower,
                "上界": upper,
                "规则": rule,
            }
        )
    return pd.DataFrame(rows)


def make_quality_summary(
    df: pd.DataFrame,
    subject_summary: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    bmi_summary: pd.DataFrame,
    sex: str,
) -> dict[str, object]:
    repeated_subjects = int(subject_summary["是否重复测量"].sum())
    repeated_rows = int(df.loc[df["是否重复测量"] == True].shape[0])
    audit = duplicate_audit.set_index("审计项")
    bmi_row = bmi_summary.iloc[0]
    return {
        "数据集": sex,
        "样本行数": int(len(df)),
        "唯一孕妇数": int(df["孕妇代码"].nunique()),
        "重复测量孕妇数": repeated_subjects,
        "重复测量涉及行数": repeated_rows,
        "完全重复行数": int(audit.loc["完全重复行", "重复组数"]),
        "同孕妇同日期重复组数": int(audit.loc["同孕妇同检测日期", "重复组数"]),
        "同孕妇同日期重复对数": int(audit.loc["同孕妇同检测日期", "重复对数"]),
        "同孕妇同孕周重复组数": int(audit.loc["同孕妇同连续孕周", "重复组数"]),
        "同孕妇同孕周重复对数": int(audit.loc["同孕妇同连续孕周", "重复对数"]),
        "同次采血重复组数": int(audit.loc["同次采血重复检测", "重复组数"]),
        "缺失单元格数": int(df.isna().sum().sum()),
        "GC异常行数": int(df["GC_abnormal"].sum()),
        "孕周异常行数": int(df["孕周异常"].sum()),
        "比例超界行数": int(df["比例变量超界"].sum()),
        "BMI可比较行数": int(bmi_row["可比较行数"]),
        "BMI差值绝对值中位数": float(bmi_row["BMI差值绝对值中位数"]),
        "BMI绝对差值超过0.5行数": int(bmi_row["绝对差值超过0.5行数"]),
        "Y相关结构性缺失行数": int(df["Y相关字段_结构性缺失"].sum()),
    }


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"找不到原始附件：{RAW_FILE}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    sheets = pd.read_excel(RAW_FILE, sheet_name=None)
    quality_tables = []
    statistics_tables = []
    statistics_by_sex: dict[str, pd.DataFrame] = {}
    category_tables = []
    duplicate_tables = []
    proportion_tables = []
    bmi_tables = []
    outlier_tables = []
    quality_summary_rows = []
    subject_tables: dict[str, pd.DataFrame] = {}
    manifest_rows = []

    source_hash = hashlib.sha256(RAW_FILE.read_bytes()).hexdigest()
    for sheet_name, raw_df in sheets.items():
        normalized = clean_column_names(raw_df)
        sex = "男胎" if (MALE_SHEET_HINT in str(sheet_name) or Y_CONCENTRATION_COLUMN in normalized.columns) else "女胎"
        prepared, source_columns, bounds = add_analysis_fields(normalized, sex)
        filename = "male_cleaned.csv" if sex == "男胎" else "female_cleaned.csv"
        prepared.to_csv(PROCESSED_DIR / filename, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

        subject_summary = make_subject_summary(prepared, sex)
        subject_filename = "subject_summary_male.csv" if sex == "男胎" else "subject_summary_female.csv"
        subject_summary.to_csv(PROCESSED_DIR / subject_filename, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        subject_tables[sex] = subject_summary

        duplicate_audit = make_duplicate_audit(normalized, prepared, sex)
        bmi_summary = make_bmi_consistency(prepared, sex)
        quality_summary_rows.append(make_quality_summary(prepared, subject_summary, duplicate_audit, bmi_summary, sex))
        quality_tables.append(make_quality_table(prepared, source_columns, sex))
        statistics = make_numeric_statistics(prepared, sex)
        statistics_tables.append(statistics)
        statistics_by_sex[sex] = statistics
        category_tables.append(make_category_counts(prepared, sex))
        duplicate_tables.append(duplicate_audit)
        proportion_tables.append(make_proportion_check(prepared, sex))
        bmi_tables.append(bmi_summary)
        outlier_tables.append(make_outlier_summary(prepared, sex, bounds))
        manifest_rows.append(
            {
                "数据集": sex,
                "工作表": sheet_name,
                "原始文件": RAW_FILE.relative_to(ROOT).as_posix(),
                "原始文件SHA256": source_hash,
                "行数": int(len(prepared)),
                "原始字段数": int(len(source_columns)),
                "输出文件": f"data/processed/{filename}",
            }
        )

    write_csv(pd.DataFrame(quality_summary_rows), "data_quality_summary.csv")
    write_csv(pd.concat(quality_tables, ignore_index=True), "missing_values.csv")
    write_csv(pd.concat(statistics_tables, ignore_index=True), "descriptive_statistics_all.csv")
    for sex, table in statistics_by_sex.items():
        filename = "descriptive_statistics_male.csv" if sex == "男胎" else "descriptive_statistics_female.csv"
        table.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")
    write_csv(pd.concat(category_tables, ignore_index=True), "category_counts.csv")
    write_csv(pd.concat(duplicate_tables, ignore_index=True), "duplicate_audit.csv")
    write_csv(pd.concat(proportion_tables, ignore_index=True), "proportion_range_check.csv")
    write_csv(pd.concat(bmi_tables, ignore_index=True), "bmi_consistency_summary.csv")
    write_csv(pd.concat(outlier_tables, ignore_index=True), "outlier_flags_summary.csv")
    write_csv(pd.DataFrame(manifest_rows), "data_manifest.csv")

    print("预处理完成：")
    for row in quality_summary_rows:
        print(
            f"{row['数据集']}：{row['样本行数']} 行，{row['唯一孕妇数']} 名孕妇，"
            f"重复测量孕妇 {row['重复测量孕妇数']} 名，缺失单元格 {row['缺失单元格数']}。"
        )


if __name__ == "__main__":
    main()
