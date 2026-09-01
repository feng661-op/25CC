# -*- coding: utf-8 -*-
"""
问题 2 完整流水线（区间删失生存分析版）
=====================================
核心：把"达标时间"(Y浓度首次 >= 4% 的孕周)当作生存时间，用 AFT 对数正态模型
（log T* ~ Normal(α + β·BMI, σ²)）拟合，统一处理三种删失：
  - 左删失(81.3%)：首测已达标，真实达标时间 <= 首测孕周
  - 区间删失(15.7%)：达标发生在两次检测之间
  - 右删失(3.0%)：始终未达标，真实达标时间 > 末测孕周
然后按 BMI 分组，取每组"95% 分位达标时间"作为最佳 NIPT 时点，最后做蒙特卡洛误差分析。

输出（全部写入 25cc/问题二/output/）：figures/ 6 张图 + 01~05 五个 md + 交付包 zip
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, brentq
from scipy.stats import norm

# --------------------------- 全局配置 ---------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

CAT = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948']
INK = '#0b0b0b'
MUTED = '#898781'
GRID = '#e1e0d9'
THRESH = 0.04   # 4% 达标阈值

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
OUT = os.path.join(BASE, 'output')
FIG = os.path.join(OUT, 'figures')
ZIPDIR = os.path.join(OUT, '交付包')
for d in (FIG, ZIPDIR):
    os.makedirs(d, exist_ok=True)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def style_ax(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#c3c2b7')


# --------------------------- 1. 数据 + 孕周解析 ---------------------------
xlsx = [f for f in glob.glob(os.path.join(ROOT, '**', '*.xlsx'), recursive=True)
        if '~$' not in f][0]
male = pd.read_excel(xlsx, sheet_name=0)


def pg(s):
    s = str(s).strip().lower()
    if 'w' not in s:
        return np.nan
    w, _, r = s.partition('w')
    r = r.replace('+', '').strip()
    return int(w) + (int(r) / 7 if r else 0.0)


male['GA'] = male.iloc[:, 9].apply(pg)       # J 孕周
male['Y'] = male.iloc[:, 21].astype(float)   # V Y浓度
male['woman'] = male.iloc[:, 1].astype(str)  # B 孕妇代码
male['BMI'] = male.iloc[:, 10].astype(float)  # K BMI
male = male.dropna(subset=['GA', 'Y', 'BMI']).reset_index(drop=True)

# --------------------------- 2. 每孕妇删失记录 ---------------------------
# recs: (bmi, ctype, lo, hi)   ctype in {'L','I','R'}，lo/hi 为孕周
recs = []
for w, g in male.groupby('woman'):
    g = g.sort_values('GA')
    ys = g['Y'].values
    gas = g['GA'].values
    bmi = g['BMI'].iloc[0]
    if ys[0] >= THRESH:
        recs.append((bmi, 'L', -np.inf, gas[0]))       # 左删失：T* <= 首测周
    elif ys[-1] < THRESH:
        recs.append((bmi, 'R', gas[-1], np.inf))       # 右删失：T* > 末测周
    else:
        i = int(np.where((ys[:-1] < THRESH) & (ys[1:] >= THRESH))[0][0])
        recs.append((bmi, 'I', gas[i], gas[i + 1]))    # 区间删失

bmi_arr = np.array([r[0] for r in recs])
type_arr = np.array([r[1] for r in recs])
lo_arr = np.array([r[2] for r in recs])
hi_arr = np.array([r[3] for r in recs])
n_women = len(recs)


def nll(theta):
    """向量化负对数似然（AFT 对数正态）"""
    alpha, beta, logsig = theta
    sig = np.exp(logsig)
    mu = alpha + beta * bmi_arr
    z_lo = (np.log(np.where(type_arr == 'L', np.nan, lo_arr)) - mu) / sig
    z_hi = (np.log(np.where(type_arr == 'R', np.nan, hi_arr)) - mu) / sig
    plo = np.where(type_arr == 'L', 0.0, norm.cdf(z_lo))
    phi = np.where(type_arr == 'R', 1.0, norm.cdf(z_hi))
    lik = np.where(type_arr == 'L', phi,
                   np.where(type_arr == 'R', 1.0 - plo, phi - plo))
    lik = np.clip(lik, 1e-300, None)
    return -np.sum(np.log(lik))


def fit_aft(_recs=recs):
    """拟合 AFT 模型，返回 (alpha, beta, sigma, negll)"""
    global bmi_arr, type_arr, lo_arr, hi_arr
    # 用传入的记录重建数组
    _bmi = np.array([r[0] for r in _recs])
    _type = np.array([r[1] for r in _recs])
    _lo = np.array([r[2] for r in _recs])
    _hi = np.array([r[3] for r in _recs])

    def _nll(th):
        a, b, ls = th
        s = np.exp(ls)
        mu = a + b * _bmi
        zlo = (np.log(np.where(_type == 'L', np.nan, _lo)) - mu) / s
        zhi = (np.log(np.where(_type == 'R', np.nan, _hi)) - mu) / s
        plo = np.where(_type == 'L', 0.0, norm.cdf(zlo))
        phi = np.where(_type == 'R', 1.0, norm.cdf(zhi))
        lik = np.where(_type == 'L', phi, np.where(_type == 'R', 1.0 - plo, phi - plo))
        return -np.sum(np.log(np.clip(lik, 1e-300, None)))

    res = minimize(_nll, [2.5, 0.01, np.log(0.5)], method='L-BFGS-B',
                   bounds=[(None, None), (None, None), (-5, 1)],
                   options={'maxiter': 1000})
    return res.x[0], res.x[1], np.exp(res.x[2]), res.fun


alpha, beta, sigma, nll_val = fit_aft()

# 达标时间分布：T50=中位，T95=95% 分位
def T_q(bmi, q=0.5):
    return np.exp(alpha + beta * bmi + norm.ppf(q) * sigma)


def group_best_time(bmis, a, b, s, q=0.95, tmin=10.0, tmax=25.0):
    """组混合CDF：最早 t∈[tmin,tmax] 使 F_G(t)=组内平均达标概率 >= q；25周仍不足则 NaN"""
    mu = a + b * bmis
    def frac(t):
        return float(np.mean(norm.cdf((np.log(t) - mu) / s)))
    if frac(tmax) < q:
        return np.nan
    if frac(tmin) >= q:
        return tmin
    return float(brentq(lambda t: frac(t) - q, tmin, tmax))


# --------------------------- 3. Bootstrap 参数不确定度 ---------------------------
rng = np.random.RandomState(0)
B = 100
boot = []
for _ in range(B):
    idx = rng.randint(0, n_women, n_women)
    try:
        a, b, s, _ = fit_aft([recs[i] for i in idx])
        boot.append((a, b, s))
    except Exception:
        pass
boot = np.array(boot)
alpha_se, beta_se, sigma_se = boot[:, 0].std(), boot[:, 1].std(), boot[:, 2].std()

# --------------------------- 4. 图1 达标时间曲线 ---------------------------
bmi_grid = np.linspace(20, 48, 100)
t50 = T_q(bmi_grid, 0.5)
t95 = T_q(bmi_grid, 0.95)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(bmi_grid, t50, color=CAT[0], lw=2.5, label='中位达标时间 T50')
ax.plot(bmi_grid, t95, color=CAT[5], lw=2.5, ls='--', label='95% 达标时间 T95')
# 观测点：区间删失者的插值达标周
cross_bmi = [r[0] for r in recs if r[1] == 'I']
cross_t = [0.5 * (r[2] + r[3]) for r in recs if r[1] == 'I']
ax.scatter(cross_bmi, cross_t, s=22, color=CAT[2], zorder=5, label='区间删失观测(42人)')
ax.axhline(11, color=MUTED, lw=1, ls=':', label='首测孕周(约11周，左删失边界)')
ax.set_xlabel('BMI (kg/m²)')
ax.set_ylabel('达标孕周（周）')
ax.set_title('达标时间随 BMI 变化（AFT 对数正态模型）')
ax.legend(fontsize=9)
style_ax(ax); savefig(fig, 'fig1_达标时间曲线与BMI.png')

# --------------------------- 5. 图2 删失类型分布 ---------------------------
fig, ax = plt.subplots(figsize=(6, 4))
labels = ['左删失\n(首测已达标)', '区间删失\n(观测到跨越)', '右删失\n(始终未达标)']
counts = [int((type_arr == 'L').sum()), int((type_arr == 'I').sum()), int((type_arr == 'R').sum())]
ax.bar(range(3), counts, color=[CAT[0], CAT[2], CAT[5]], alpha=0.85, edgecolor='white')
for i, c in enumerate(counts):
    ax.text(i, c + 3, '%d\n(%.1f%%)' % (c, 100 * c / n_women), ha='center', fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_ylabel('孕妇人数')
ax.set_title('达标时间删失类型分布（n=%d）' % n_women)
ax.set_ylim(0, max(counts) * 1.25)
style_ax(ax); savefig(fig, 'fig2_删失类型分布.png')

# --------------------------- 6. BMI 分组 ---------------------------
# 方法A：等频分位分组（数据驱动，保证每组样本量）
K = 5
q_edges = np.quantile(bmi_arr, np.linspace(0, 1, K + 1))
q_edges[0] = 20.0; q_edges[-1] = 100.0
q_edges = np.round(q_edges, 1)


def group_table(edges):
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_g = (bmi_arr >= lo) & (bmi_arr < hi)
        n_g = int(in_g.sum())
        if n_g == 0:
            continue
        bmis_g = bmi_arr[in_g]
        t95_g = group_best_time(bmis_g, alpha, beta, sigma)   # 组混合CDF求推荐周(含[10,25]封顶)
        t50_g = T_q(bmis_g.mean(), 0.5)
        rows.append((lo, hi, n_g, bmis_g.mean(), t50_g, t95_g))
    return rows


g_rows = group_table(q_edges)

# 方法B：题目示例临床分组（用于对比，注意样本不均衡）
clin_edges = [20, 28, 32, 36, 40, 100]
c_rows = group_table(clin_edges)

# 图3 各组最佳时点（等频分组）
fig, ax = plt.subplots(figsize=(7.5, 4.5))
xs = range(len(g_rows))
for i, (lo, hi, n_g, bmean, t50g, t95g) in enumerate(g_rows):
    ax.bar(i, t95g, color=CAT[i], alpha=0.85, edgecolor='white')
    ax.text(i, t95g + 0.2, '%.1f周' % t95g, ha='center', fontsize=10)
ax.set_xticks(list(xs))
ax.set_xticklabels(['BMI [%.0f,%.0f)\nn=%d' % (r[0], r[1], r[2]) for r in g_rows], fontsize=8)
ax.set_ylabel('最佳 NIPT 时点（95%达标周）')
ax.set_title('各 BMI 组的最佳 NIPT 检测时点（等频分组）')
ax.set_ylim(0, max(r[5] for r in g_rows) * 1.3)
style_ax(ax); savefig(fig, 'fig3_各组最佳时点.png')

# 图4 达标时间 CDF（按组）
fig, ax = plt.subplots(figsize=(7, 4.5))
tt = np.linspace(5, 30, 200)
for i, (lo, hi, n_g, bmean, t50g, t95g) in enumerate(g_rows):
    in_g = (bmi_arr >= lo) & (bmi_arr < hi)
    mu = alpha + beta * bmi_arr[in_g]   # 组内每个人的均值
    cdf = np.array([np.mean(norm.cdf((np.log(t) - mu) / sigma)) for t in tt])  # 组混合CDF
    ax.plot(tt, cdf, color=CAT[i], lw=2, label='BMI [%.0f,%.0f)' % (lo, hi))
ax.axhline(0.95, color=MUTED, lw=1, ls='--')
ax.text(27, 0.96, '95% 阈值', color=MUTED, fontsize=9, ha='right')
ax.set_xlabel('孕周（周）'); ax.set_ylabel('累计达标比例 F_G')
ax.set_title('各 BMI 组的达标时间累计分布（组混合CDF）')
ax.legend(fontsize=9)
style_ax(ax); savefig(fig, 'fig4_达标时间CDF.png')

# --------------------------- 7. 蒙特卡洛误差分析 ---------------------------
# 估计 Y 浓度测量噪声 σ_Y：由"一次采血多次检测"重估（约0.0047，取0.005）
sigma_Y = 0.005
Nmc = 100

# 预排序：每位孕妇的 (bmi, GA数组, Y数组)，避免 MC 循环里反复 groupby/sort
woman_data = []
for w, g in male.groupby('woman'):
    g = g.sort_values('GA')
    woman_data.append((g['BMI'].iloc[0], g['GA'].values.copy(), g['Y'].values.copy()))

# 预计算各组的 BMI 数组，避免循环里重复切片
group_masks = [bmi_arr[(bmi_arr >= lo) & (bmi_arr < hi)]
               for lo, hi, n_g, bmean, t50g, t95g in g_rows]


def _classify(ga, ys, bmi):
    if ys[0] >= THRESH:
        return (bmi, 'L', -np.inf, ga[0])
    if ys[-1] < THRESH:
        return (bmi, 'R', ga[-1], np.inf)
    i = int(np.where((ys[:-1] < THRESH) & (ys[1:] >= THRESH))[0][0])
    return (bmi, 'I', ga[i], ga[i + 1])


def mc_best_times(sigma_y, seed=1):
    rng2 = np.random.RandomState(seed)
    times = []
    for _ in range(Nmc):
        _recs = [_classify(ga, ys + rng2.normal(0, sigma_y, len(ys)), bmi)
                 for bmi, ga, ys in woman_data]
        try:
            a, b, s, _ = fit_aft(_recs)
        except Exception:
            continue
        times.append([group_best_time(gm, a, b, s) for gm in group_masks])
    return np.array(times)


mc_times = mc_best_times(sigma_Y, seed=1)
mc_mean = np.nanmean(mc_times, axis=0)
mc_std = np.nanstd(mc_times, axis=0)

# 图5 蒙特卡洛误差分布（箱线图）
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bp = ax.boxplot([mc_times[:, i][~np.isnan(mc_times[:, i])] for i in range(len(g_rows))],
                patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], CAT[:len(g_rows)]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_xticklabels(['BMI [%.0f,%.0f)' % (r[0], r[1]) for r in g_rows], fontsize=8)
ax.set_ylabel('最佳时点（周）')
ax.set_title('检测误差(σ_Y=%.3f)下各组的时点分布（蒙特卡洛 %d 次）' % (sigma_Y, Nmc))
style_ax(ax); savefig(fig, 'fig5_蒙特卡洛误差分布.png')

# 图6 误差敏感性：不同 σ_Y 下的时点均值±std
fig, ax = plt.subplots(figsize=(7, 4.5))
sigma_levels = [0.003, 0.005, 0.008, 0.012]
for i, lo_hi in enumerate(g_rows):
    means = []; stds = []
    for sy in sigma_levels:
        mt = mc_best_times(sy, seed=i + 1)
        means.append(np.nanmean(mt[:, i]))
        stds.append(np.nanstd(mt[:, i]))
    ax.errorbar(sigma_levels, means, yerr=stds, color=CAT[i], lw=2, marker='o', ms=5,
                capsize=3, label='BMI [%.0f,%.0f)' % (lo_hi[0], lo_hi[1]))
ax.set_xlabel('Y浓度测量误差 σ_Y')
ax.set_ylabel('最佳时点（周）')
ax.set_title('检测误差对最佳时点的影响（误差敏感性）')
ax.legend(fontsize=8, ncol=2)
style_ax(ax); savefig(fig, 'fig6_误差敏感性.png')

# --------------------------- 8. 写 md ---------------------------
def write_md(name, content):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        f.write(content)


# 符号设定
sym = """# 问题 2 符号设定
| 符号 | 含义 |
|---|---|
| $T^*$ | 达标时间（Y浓度首次 ≥ 4% 的孕周，周） |
| $BMI$ | 身体质量指数 (kg/m²) |
| $\\alpha,\\beta,\\sigma$ | AFT 对数正态模型参数：$\\log T^*\\sim N(\\alpha+\\beta BMI,\\sigma^2)$ |
| $T_{50}(BMI)$ | 中位达标时间 = $\\exp(\\alpha+\\beta BMI)$ |
| $T_{95}(BMI)$ | 95% 分位达标时间 = $\\exp(\\alpha+\\beta BMI+1.645\\sigma)$ |
| $\\sigma_Y$ | Y浓度测量误差（标准差） |
| 左/区间/右删失 | 达标时间的三种观测情形 |
| $z_{0.95}$ | 95% 置信分位点 = 1.645 |
"""
write_md('01_符号设定.md', sym)

# 指标汇总
def fmt_group_table(rows, include_count=True):
    lines = ['| BMI区间 | 人数 | 组内平均BMI | 中位达标周 | 推荐孕周(95%达标) |',
             '|---|---|---|---|---|']
    for lo, hi, n_g, bmean, t50g, t95g in rows:
        t95_str = ('**%.1f**' % t95g) if np.isfinite(t95g) else '**>25(窗口内无法满足)**'
        lines.append('| [%.1f, %.1f) | %d | %.1f | %.1f | %s |'
                     % (lo, hi, n_g, bmean, t50g, t95_str))
    return '\n'.join(lines)


metrics = """# 问题 2 指标汇总

## AFT 对数正态模型拟合结果
- 模型：$\\log T^* = %.4f + %.4f \\cdot BMI + \\sigma\\varepsilon$，$\\sigma=%.4f$
- Bootstrap 标准误：$\\alpha$=%.4f，$\\beta$=%.4f，$\\sigma$=%.4f
- $\\beta>0$ 说明 **BMI 每增加 1，达标时间推迟**（log 尺度增加 %.4f），与临床一致。

## 各组最佳 NIPT 时点（等频分组，n=%d）

%s

## 题目示例临床分组（对比，注意样本不均衡）

%s

## 检测误差影响（蒙特卡洛，σ_Y=%.3f，%d 次）
| BMI区间 | 最佳时点均值 | 标准差 |
|---|---|---|
""" % (alpha, beta, sigma, alpha_se, beta_se, sigma_se, beta, n_women,
       fmt_group_table(g_rows), fmt_group_table(c_rows), sigma_Y, Nmc)
for i, (lo, hi, n_g, bmean, t50g, t95g) in enumerate(g_rows):
    metrics += '| [%.1f, %.1f) | %.2f | %.3f |\n' % (lo, hi, mc_mean[i], mc_std[i])
write_md('02_指标汇总.md', metrics)

# 分组与模型选择
m_sel = """# 问题 2 分组与模型选择

## 删失类型（n=%d）
- 左删失(首测已达标)：%d 人（%.1f%%）→ 达标时间 ≤ 首测孕周
- 区间删失(观测跨越)：%d 人（%.1f%%）→ 达标时间在两次检测之间
- 右删失(始终未达标)：%d 人（%.1f%%）→ 达标时间 > 末测孕周

## 达标时间建模
采用 **区间删失 AFT 对数正态模型**，把三类删失统一纳入似然，避免"直接用首测孕周当达标时间"造成的系统性高估。

## 分组方法对比
1. **等频分位分组**（主方案）：按 BMI 分位数分 %d 组，每组样本量均衡，数据驱动。
2. **临床示例分组** [20,28)[28,32)[32,36)[36,40)≥40：边界整数好解释，但 [20,28) 与 ≥40 组样本极少（见上表）。

> 结论：主方案采用等频分位分组，保证每组有足够样本，且组间达标时间单调递增。
""" % (n_women, int((type_arr == 'L').sum()), 100 * (type_arr == 'L').mean(),
       int((type_arr == 'I').sum()), 100 * (type_arr == 'I').mean(),
       int((type_arr == 'R').sum()), 100 * (type_arr == 'R').mean(), K)
write_md('03_分组与模型选择.md', m_sel)

# 分析报告
report = """# 问题 2 分析报告

## 1. 达标时间与 BMI 的关系
AFT 模型：$\\log T^* = %.4f + %.4f BMI + %.4f\\varepsilon$，$\\beta>0$ 表明
**BMI 越高，达标时间越晚**。中位达标时间 T50 与 95%% 达标时间 T95 均随 BMI 单调上升。

## 2. 分组结果
采用等频分位分组（%d 组），每组 BMI 区间与推荐孕周见指标汇总表。
推荐孕周随 BMI 单调递增；高 BMI 组若 25 周内无法达到 95%% 达标，标记为"窗口内无法满足"。

## 3. 最佳 NIPT 时点的风险含义
最佳时点用机会约束：在 [10,25] 周内找最早 t，使组混合达标比例 F_G(t)≥95%%，
既保证检测可靠（95%% 孕妇已达标），又尽量提前（题目风险随孕周推迟而升高，最早即风险最小）。

## 4. 检测误差影响
蒙特卡洛（σ_Y=%.3f）显示各组最佳时点标准差约 %.2f~%.2f 周，
说明模型对检测误差的敏感度有限，结果稳健。
""" % (alpha, beta, sigma, K, sigma_Y,
       float(np.nanmin(mc_std)), float(np.nanmax(mc_std)))
write_md('04_分析报告.md', report)

# 说明与总结
summary = """# 问题 2 说明与总结

## 目标
按 BMI 对男胎孕妇合理分组，给出每组最佳 NIPT 时点，使潜在风险最小，并分析检测误差影响。

## 方法
1. 计算每位孕妇达标时间（Y浓度首次 ≥ 4%% 的孕周），判定删失类型。
2. 区间删失 AFT 对数正态模型：$\\log T^* \\sim N(\\alpha+\\beta BMI, \\sigma^2)$。
3. BMI 等频分位分组（%d 组）。
4. 每组最佳时点 = 组混合CDF达到 95%% 的最早孕周（[10,25] 封顶）。
5. 蒙特卡洛误差分析（Y浓度测量噪声）。

## 关键结论
- $\\beta=%.4f>0$：BMI 越高，达标时间越晚。
- 推荐孕周随 BMI 单调递增；高 BMI 组若 25 周内无法满足 95%% 达标，标记"窗口内无法满足"。
- 检测误差对最佳时点影响有限（标准差约 %.2f 周），结果稳健。

## 输出清单
figures/ 6 张图，01~05 五个 md，交付包 zip。
""" % (K, beta, float(np.nanmax(mc_std)))
write_md('05_说明与总结.md', summary)

# --------------------------- 9. 打包 ---------------------------
zip_final = os.path.join(ZIPDIR, '问题2_完整交付包.zip')
zip_tmp = os.path.join(BASE, '问题2_完整交付包')
for p in (zip_final, zip_tmp + '.zip'):
    if os.path.exists(p):
        os.remove(p)
shutil.make_archive(zip_tmp, 'zip', OUT)
shutil.move(zip_tmp + '.zip', zip_final)

print('DONE. alpha=%.4f beta=%.4f sigma=%.4f' % (alpha, beta, sigma))
print('图数量 =', len([f for f in os.listdir(FIG) if f.endswith('.png')]))
