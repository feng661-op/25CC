# 2025 C题 NIPT 数据预处理与探索性分析

## 1 数据来源与可复现性

本报告由 src/preprocess.py 和 src/eda.py 从原始 Excel 重新生成。原始文件为 2025C原题/附件.xlsx，使用工作表为 男胎检测数据、女胎检测数据；原始文件 SHA-256 为 14827156218bd4f7e4f16db4aa6d9f757c6648379e038ae6c6b58383648614af。仓库中的 data/raw/附件.xlsx 只作镜像，未作为独立分析来源。

运行环境记录见 [runtime_environment.txt](outputs/tables/runtime_environment.txt)，原始来源和哈希记录见 [data_manifest.csv](outputs/tables/data_manifest.csv)。本报告只覆盖数据预处理、质量审计、描述性统计、重复测量和 EDA，不包含问题一至问题四的最终模型。

## 2 数据规模

这里同时报告记录层面和孕妇层面。记录不是相互独立的孕妇，后续统计应识别同一孕妇内的重复记录。

| 数据集 | 记录数 | 孕妇数 | 重复测量孕妇数 | 重复测量比例 |
|---|---:|---:|---:|---:|
| 男胎 | 1082 | 267 | 260 | 97.4% |
| 女胎 | 605 | 147 | 143 | 97.3% |

男胎每名孕妇记录数为 1–8 条，中位数 4 条；女胎为 1–9 条，中位数 4 条。男胎约 97.4%、女胎约 97.3% 的孕妇存在重复记录，说明 record-level 与 subject-level 必须分开解释。

## 3 数据质量审计

### 3.1 原始字段、缺失与结构性缺失

- 只清理列名首尾空格，原始字段和值均保留；处理后另行追加连续孕周、BMI、阈值和重复测量分析字段。
- 孕周按“周数 + 天数 / 7”换算，例如 11w+6 为 11.857142... 周；男胎解析失败 0 条，女胎解析失败 0 条。失败数量保留在质量表中，没有静默转成数字。
- 处理后男胎表缺失单元格 980 个，女胎表缺失单元格 3573 个。女胎 Y 染色体 Z 值和 Y 染色体浓度是结构性缺失，不填 0，也不做均值填补。
- BMI_calc 按身高和体重重算；男胎 BMI 差值绝对值中位数为 0.000000，女胎为 0.000000。所有有效 BMI_calc 均通过合理范围检查。
- GC_range_flag 表示 GC 不在题面经验区间 0.40–0.60 内的记录标记：男胎 451 条，女胎 220 条；该标记不等于测序失败。
- 比例变量、孕周范围、IQR 极端值和 BMI 一致性均只做审计或标记，不暴力删除。明细见 [missing_values.csv](outputs/tables/missing_values.csv)、[proportion_range_check.csv](outputs/tables/proportion_range_check.csv)、[bmi_consistency_summary.csv](outputs/tables/bmi_consistency_summary.csv) 和 [outlier_flags_summary.csv](outputs/tables/outlier_flags_summary.csv)。

缺失率：[01_missing_rate.png](outputs/figures/01_missing_rate.png)；BMI：[02_bmi_distribution.png](outputs/figures/02_bmi_distribution.png)；孕周：[03_gestational_week_distribution.png](outputs/figures/03_gestational_week_distribution.png)。

## 4 数据层级结构

男胎存在 260 名重复测量孕妇，占孕妇人数 97.4%；女胎存在 143 名重复测量孕妇，占孕妇人数 97.3%。这说明绝大多数孕妇存在重复记录，record-level 记录不等于 subject-level 独立样本。

完整重复行、同孕妇同日期、同孕妇同孕周和同次采血的审计见 [duplicate_audit.csv](outputs/tables/duplicate_audit.csv)。男胎同次采血重复组数为 40，女胎对应审计也保留。

同一孕妇多次出现意味着 1082 条男胎记录不等于 267 个独立孕妇观测，605 条女胎记录也不等于 147 个独立孕妇观测。后续训练和验证应按孕妇分组；本报告中的普通 P 值只作为记录层面探索参考。

## 5 男胎 EDA

### 4.1 Y 浓度与 4% 阈值

男胎 Y 浓度观测范围为 0.0100–0.2342。按 Y≥0.04 定义，1082 条记录中 937 条达标（86.6%），145 条未达标。该比例是记录层面比例，不是孕妇层面比例。

[04_y_distribution_threshold.png](outputs/figures/04_y_distribution_threshold.png)

### 4.2 Y 浓度与连续孕周

记录层面 Spearman ρ=0.084，普通 P=0.006，n=1082 records。该普通 P 把每条记录视作行级观测，不能当作独立孕妇显著性检验依据。

按孕妇整簇重采样的 cluster bootstrap 结果为：点估计 0.084，中位数 0.085，95% CI [0.012, 0.161]，有效迭代 2000/2000。个体内简单斜率的正斜率比例为 87.3%，可估计斜率孕妇数为 260。

因此，当前样本只能表述为记录层面与孕周存在正向单调关联，cluster bootstrap 给出了受试者内相关下的不确定性；总体相关系数和个体内纵向方向不是同一个统计量。[05_y_vs_week.png](outputs/figures/05_y_vs_week.png)

### 4.3 Y 浓度与 BMI

Y 与检测时 BMI（measurement-time BMI）的记录层面 Spearman ρ=-0.155，普通 P=<0.001，n=1082 records。按孕妇整簇重采样后，95% CI 为 [-0.254, -0.049]。

控制连续孕周后的部分 Spearman 点估计为 -0.169，cluster bootstrap 95% CI 为 [-0.270, -0.061]。这些结果是描述性关联，不能解释为 BMI 导致 Y 浓度变化。[06_y_vs_measurement_bmi.png](outputs/figures/06_y_vs_measurement_bmi.png)

### 4.4 孕周 × baseline BMI

BMI 分组严格按孕妇级 baseline BMI：先取每名孕妇最早连续孕周的 BMI_calc；若最早孕周有多条记录则取其中位数，再传播到该孕妇的所有记录。当前采用“等频分组”，每个组的孕妇人数和记录数见 [bmi_group_definitions.csv](outputs/tables/bmi_group_definitions.csv)。分组只用于 EDA，不代表问题二最终最优 BMI 分组。

[07_week_baseline_bmi_pass_heatmap.png](outputs/figures/07_week_baseline_bmi_pass_heatmap.png)

热力图的 BMI 轴使用孕妇级 baseline BMI 分组，格内达标率仍是记录层面观测比例，并同时保留记录数和孕妇数；记录数小于5的格子不显示。[08_pass_rate_by_baseline_bmi.png](outputs/figures/08_pass_rate_by_baseline_bmi.png)

BMI 组与首次观测达标的描述如下：

- 等频组1（20.7–29.4）：54 名孕妇，观察到达标 54 名（100.0%）；首次观测达标孕周中位数 12.86 周，未观测达标比例 0.0%。
- 等频组2（29.4–30.5）：54 名孕妇，观察到达标 53 名（98.1%）；首次观测达标孕周中位数 12.71 周，未观测达标比例 1.9%。
- 等频组3（30.5–32.2）：52 名孕妇，观察到达标 51 名（98.1%）；首次观测达标孕周中位数 12.71 周，未观测达标比例 1.9%。
- 等频组4（32.2–33.9）：53 名孕妇，观察到达标 51 名（96.2%）；首次观测达标孕周中位数 12.71 周，未观测达标比例 3.8%。
- 等频组5（33.9–46.9）：54 名孕妇，观察到达标 51 名（94.4%）；首次观测达标孕周中位数 13.29 周，未观测达标比例 5.6%。

### 4.5 重复测量与个体内方向

在至少两条记录的男胎孕妇中，个体内 Y 标准差中位数为 0.0121，个体内极差中位数为 0.0276。按每名孕妇内部 Y—孕周简单线性斜率统计：正斜率 227 人（87.3%），负斜率 33 人（12.7%），0附近斜率 17 人（6.5%）。0附近定义为斜率绝对值不超过 0.0005。斜率只是个体内描述，不是最终线性模型。

[09_repeated_measurement_trajectory.png](outputs/figures/09_repeated_measurement_trajectory.png) 展示记录次数较多孕妇的轨迹；[13_male_subject_slope_distribution.png](outputs/figures/13_male_subject_slope_distribution.png) 展示斜率分布。近似同孕周重复对另存于 [near_week_repeat_pairs.csv](outputs/tables/near_week_repeat_pairs.csv)，不与同次采血重复检测混用。

### 4.6 阈值观测与删失

“首次观测达标孕周”只表示第一次观测到 Y≥4%，不是无删失的真实跨阈值时刻。每名男胎孕妇恰好一行的阈值表见 [male_threshold_censoring.csv](outputs/tables/male_threshold_censoring.csv)，类型统计见 [threshold_censoring_summary.csv](outputs/tables/threshold_censoring_summary.csv)。

- left：217 名（81.3%），只能知道真实跨越时间不晚于首次观测。
- interval：43 名（16.1%），跨越时间位于最后一次未达标和首次达标之间。
- right：7 名（2.6%），截至最后观测仍未观察到达标。
- 非单调轨迹：37 名（13.9%）；这是独立标志，与三类删失不互斥。

按 baseline BMI 分组的删失类型统计见 [threshold_censoring_by_bmi.csv](outputs/tables/threshold_censoring_by_bmi.csv)，图 14 同时展示总体类型和组内类型比例。

[10_first_observed_pass_week_distribution.png](outputs/figures/10_first_observed_pass_week_distribution.png)；[11_first_observed_pass_week_vs_baseline_bmi.png](outputs/figures/11_first_observed_pass_week_vs_baseline_bmi.png)；[14_male_threshold_censoring_types.png](outputs/figures/14_male_threshold_censoring_types.png)。

### 4.7 同次采血重复检测误差

男胎同次采血重复检测单独按孕妇代码和检测抽血次数分组，共 40 个重复组、82 个重复对。Y 绝对差中位数为 0.004873，95 百分位数为 0.016027；相对差中位数为 9.7%，95 百分位数为 41.0%。差值和 Bland–Altman 描述性界限见 [male_same_draw_repeat_error_summary.csv](outputs/tables/male_same_draw_repeat_error_summary.csv)。

A/B 按序号确定性排序；同一次采血没有天然先后测量方向，差值符号只作描述，绝对差更加直接。不要把这部分与近似同孕周重复对混为一谈。[15_male_same_draw_repeat_error.png](outputs/figures/15_male_same_draw_repeat_error.png)

### 4.8 测序质量与 Y

与 Y 浓度绝对值关联最大的质量变量是 比对比例，记录层面 Spearman ρ=-0.159，普通探索性 P=<0.001，n=1082 records。完整表见 [quality_vs_y_summary.csv](outputs/tables/quality_vs_y_summary.csv)，图见 [16_quality_vs_y.png](outputs/figures/16_quality_vs_y.png)。

这些质量变量与 Y 存在统计关联，因此后续可以把它们作为质量协变量或敏感性分析变量；当前 EDA 不据此作因果判断。

候选变量的记录层面关联排序如下：

- X 染色体浓度（比例值）：记录层面 Spearman ρ=0.474；普通探索性 P=<0.001，n=1082 records。
- 检测抽血次数：记录层面 Spearman ρ=0.323；普通探索性 P=<0.001，n=1082 records。
- 18 号染色体 Z 值：记录层面 Spearman ρ=-0.178；普通探索性 P=<0.001，n=1082 records。
- 体重（kg）：记录层面 Spearman ρ=-0.167；普通探索性 P=<0.001，n=1082 records。
- 比对比例：记录层面 Spearman ρ=-0.159；普通探索性 P=<0.001，n=1082 records。
- 检测时 BMI（kg/m²）：记录层面 Spearman ρ=-0.155；普通探索性 P=<0.001，n=1082 records。
- 年龄（岁）：记录层面 Spearman ρ=-0.117；普通探索性 P=<0.001，n=1082 records。
- Y 染色体 Z 值：记录层面 Spearman ρ=0.113；普通探索性 P=<0.001，n=1082 records。

## 6 女胎 EDA

### 5.1 记录级异常分布

女胎共有 605 条 records，其中任意异常 67 条（11.1%），正常 538 条。该比例是异常记录比例。[17_female_abnormal_distribution.png](outputs/figures/17_female_abnormal_distribution.png)

记录级类型统计见 [female_abnormal_counts.csv](outputs/tables/female_abnormal_counts.csv)：正常 538 条；T13 10 条；T18 33 条；T21 9 条；复合异常 15 条。

### 5.2 孕妇级异常分布

147 名孕妇中，孕妇级统计为：任意异常 44 名（29.9%）；T13 18 名（12.2%）；T18 30 名（20.4%）；T21 12 名（8.2%）；复合异常 12 名（8.2%）。孕妇级结果见 [female_subject_abnormal_summary.csv](outputs/tables/female_subject_abnormal_summary.csv) 和 [female_subject_abnormal_counts.csv](outputs/tables/female_subject_abnormal_counts.csv)。

记录级异常数与孕妇级异常数不能互相替代；同一孕妇只要任一检测记录出现对应标志，孕妇级标志就记为1。记录级任意异常为 67 条，孕妇级任意异常为 44 名（29.9%）。

### 5.3 同一孕妇标签一致性

标签一致性按每名孕妇的异常类型集合审计，不决定哪个检测标签是真值。共 43 名孕妇（29.3%）存在不同记录标签，细节见 [female_within_subject_label_consistency.csv](outputs/tables/female_within_subject_label_consistency.csv)，汇总见 [female_within_subject_label_consistency_summary.csv](outputs/tables/female_within_subject_label_consistency_summary.csv)。

### 5.4 Z13、Z18、Z21 的两种合法描述

对于每个对应异常，均同时给出原始方向 Z 的 AUC 和绝对值 Z 的 AUC；二者不择一。P 值是记录层面探索性 Mann–Whitney U P 值，存在重复测量，不能解释成独立孕妇检验。

- 13 号染色体 Z 值（T13异常）：raw Z AUC=0.420，absolute Z AUC=0.473；Mann–Whitney U 的记录层面探索性 P=0.192。
- 18 号染色体 Z 值（T18异常）：raw Z AUC=0.543，absolute Z AUC=0.515；Mann–Whitney U 的记录层面探索性 P=0.338。
- 21 号染色体 Z 值（T21异常）：raw Z AUC=0.515，absolute Z AUC=0.495；Mann–Whitney U 的记录层面探索性 P=0.851。

[18_z13_vs_t13.png](outputs/figures/18_z13_vs_t13.png)　[19_z18_vs_t18.png](outputs/figures/19_z18_vs_t18.png)　[20_z21_vs_t21.png](outputs/figures/20_z21_vs_t21.png)

X 染色体 Z 值与任意异常的辅助图见 [21_zx_vs_abnormal.png](outputs/figures/21_zx_vs_abnormal.png)。完整新版结果见 [female_zscore_discrimination_summary.csv](outputs/tables/female_zscore_discrimination_summary.csv)。

### 5.5 孕妇级 Z 值辅助描述与其他特征

每名女胎孕妇的 max_abs_Z13、max_abs_Z18、max_abs_Z21、max_abs_ZX 和对应中位数见 [female_subject_z_summary.csv](outputs/tables/female_subject_z_summary.csv)。这只是孕妇级描述性聚合，不是最终分类器表现。

其他候选变量的记录层面描述同时保留原始方向 AUC 和绝对值 AUC，不使用单一指标作最终特征筛选：

- 检测时 BMI（kg/m²）：正常中位数 31.445，异常中位数 31.641；原始方向 AUC=0.468，绝对值 AUC=0.468。
- 连续孕周（周）：正常中位数 17.429，异常中位数 19.857；原始方向 AUC=0.559，绝对值 AUC=0.559。
- GC 含量（比例）：正常中位数 0.401，异常中位数 0.401；原始方向 AUC=0.479，绝对值 AUC=0.479。
- 原始读段数（条）：正常中位数 4588871.500，异常中位数 4478682.000；原始方向 AUC=0.488，绝对值 AUC=0.488。
- 比对比例：正常中位数 0.801，异常中位数 0.802；原始方向 AUC=0.504，绝对值 AUC=0.504。
- 重复读段比例：正常中位数 0.030，异常中位数 0.030；原始方向 AUC=0.469，绝对值 AUC=0.469。
- 过滤读段比例：正常中位数 0.023，异常中位数 0.023；原始方向 AUC=0.483，绝对值 AUC=0.483。

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

图表为统一中文风格，PNG 由脚本以 300 dpi 保存；本次字体回退为 Microsoft YaHei。完整数字来自脚本运行时的表格，不手工固定样本量。

> 本报告严格限定于原始数据审计、预处理、数据质量检查、特征整理、重复测量描述、检测误差描述、描述性统计、可视化和 EDA；未修改问题一最终模型，也未完成问题二、问题三或问题四最终模型。
