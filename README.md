# 2025 年全国大学生数学建模竞赛 C 题：NIPT 数据分析与建模

仓库包含原始数据审计、可追溯预处理、数据质量检查、探索性分析，以及分问题的正式建模实现。问题四已经形成独立的女胎 T13/T18/T21 异常综合判定流水线；其余问题继续以各自目录中的实现为准。

## 数据来源

仓库内的 2025C原题/附件.xlsx 是分析使用的原始附件；data/raw/附件.xlsx 是同 SHA-256 的镜像。来源、工作表、文件哈希和记录数会写入 outputs/tables/data_manifest.csv。原始附件不会被脚本改写。

## 运行方法

在仓库根目录执行：

```bash
python src/preprocess.py
python src/eda.py
python -m py_compile src/preprocess.py src/eda.py
```

依赖版本要求见 requirements.txt，本次实际运行环境见 outputs/tables/runtime_environment.txt。脚本使用仓库相对路径，重复运行会覆盖同名产物，不会追加重复记录。

### 问题四正式建模

```bash
D:\mypython\python.exe 问题四\code\q4_pipeline.py
```

问题四以女胎 AB 列为 T13/T18/T21 三个二元监督标签；每条检测记录做预测，但外层/内层交叉验证均以孕妇整体分组。流水线比较单 |Z| 基准、染色体信息 Ridge 和 16 特征综合 Ridge，使用 PR-AUC 选超参数、训练层内 F1 阈值、孕妇级 cluster bootstrap 95% CI，并附 F2、X 染色体浓度和测序质量变量敏感性分析。正式结果、逐记录 OOF/最终判定和图表位于 `问题四/output/`。

## 目录结构

- 2025C原题/：原始题目附件。
- data/raw/：原始附件镜像。
- data/processed/：保留原始字段并追加分析字段的男胎、女胎记录表和孕妇汇总表。
- outputs/tables/：质量审计、描述统计、聚类 bootstrap、BMI 分组、阈值删失、重复检测误差和女胎孕妇级汇总。
- outputs/figures/：连续唯一编号、中文标注、300 dpi 的 PNG 图表。
- EDA_REPORT.md：由脚本根据实际表格数字自动生成的 EDA 报告。

## 重要处理原则

- 孕周按“周数 + 天数 / 7”转换为连续周数；BMI 同时保留计算值和原始差值。
- measurement_BMI 表示每条检测记录当时的 BMI；baseline_BMI 表示该孕妇最早检测孕周内 BMI 的中位数。两者不能混用。
- 男胎 Y≥4% 只生成记录级达标标志；女胎 Y 相关列是结构性缺失，不填 0。
- 异常值、GC 经验范围标记和比例越界只审计或标记，不批量删除。GC 不在 0.40–0.60 内只表示范围标记，不等于测序失败。
- 原始 Pearson、Spearman P 值只作记录层面探索描述；主要关联另给按孕妇整簇重采样的 95% 置信区间。
- 同一孕妇的记录不是独立样本。训练、验证和交叉验证应按孕妇分组；后验健康状态不能作为异常预测输入。

## 关键交付

- cluster_bootstrap_correlations.csv：Y 与孕周、检测时 BMI 及控制孕周后的部分 Spearman 的孕妇级聚类 bootstrap 区间。
- male_subject_slope_summary.csv：个体内 Y—孕周方向的描述性汇总。
- male_threshold_censoring.csv：每名男胎孕妇一行的 left、interval、right 观测删失区间，并单独标记非单调轨迹。
- threshold_censoring_by_bmi.csv：按孕妇级 baseline_BMI 汇总的阈值删失类型。
- male_same_draw_repeat_groups.csv、male_same_draw_repeat_pairs.csv、male_same_draw_repeat_error_summary.csv：同次采血重复检测误差描述。
- female_zscore_discrimination_summary.csv：Z13、Z18、Z21 的原始方向 AUC 与绝对值 AUC，P 值标明为记录层面探索性结果。
- female_subject_abnormal_summary.csv、female_subject_abnormal_counts.csv、female_within_subject_label_consistency.csv、female_subject_z_summary.csv：女胎孕妇级异常、标签一致性和 Z 值辅助描述。
- variable_dictionary.csv：关键变量定义、层级、来源、计算方式和使用边界。

## 两个使用警告

1. “首次观测达标孕周”不是无删失的真实阈值跨越时间；应结合阈值区间和删失类型解释。
2. BMI 分组只用于 EDA，按孕妇级 baseline_BMI 建立，不代表问题二的最终最优分组。

完整结论、限制和图表推荐见 EDA_REPORT.md。
