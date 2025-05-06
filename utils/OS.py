import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# 加载数据
train_df = pd.read_csv("train.csv")
internal_df = pd.read_csv("internal_test.csv")
external_df = pd.read_csv("external_test.csv")


def analyze_dataset(df, label):
    print(f"\n===== {label} =====")

    # 中位生存期
    kmf0 = KaplanMeierFitter()
    kmf1 = KaplanMeierFitter()

    df_0 = df[df["MVI"] == 0]
    df_1 = df[df["MVI"] == 1]

    kmf0.fit(df_0["time"], event_observed=df_0["status"], label="MVI = 0")
    kmf1.fit(df_1["time"], event_observed=df_1["status"], label="MVI = 1")

    median_0 = kmf0.median_survival_time_
    median_1 = kmf1.median_survival_time_

    print(f"中位生存期 (MVI=0): {median_0:.1f} 月")
    print(f"中位生存期 (MVI=1): {median_1:.1f} 月")

    # Log-rank 检验
    logrank_result = logrank_test(df_0["time"], df_1["time"],
                                  event_observed_A=df_0["status"],
                                  event_observed_B=df_1["status"])
    p_value = logrank_result.p_value

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    kmf0.plot(ax=ax, ci_show=True, ci_alpha=0.3, color='blue', linewidth=2,
              show_censors=True, censor_styles={'marker': '|', 'ms': 10, 'mew': 1.5})
    kmf1.plot(ax=ax, ci_show=True, ci_alpha=0.3, color='red', linewidth=2,
              show_censors=True, censor_styles={'marker': '|', 'ms': 10, 'mew': 1.5})

    # 中位生存线
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    if not pd.isna(median_0):
        plt.axvline(x=median_0, color='blue', linestyle=':', alpha=0.5)
    if not pd.isna(median_1):
        plt.axvline(x=median_1, color='red', linestyle=':', alpha=0.5)

    # 注释和美化
    plt.title(f"{label} Kaplan-Meier Survival Curve", fontsize=14)
    plt.xlabel("Time (months)", fontsize=12)
    plt.ylabel("Survival probability", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='MVI Status', fontsize=10, title_fontsize=11)

    if not pd.isna(median_0):
        plt.text(median_0 + 2, 0.2, f'Median: {median_0:.1f} months', color='blue', ha='left')
    if not pd.isna(median_1):
        plt.text(median_1 + 2, 0.2, f'Median: {median_1:.1f} months', color='red', ha='left')

    plt.text(0.2, 0.2, f'Log-rank p = {p_value:.3f}',
             transform=ax.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


    add_at_risk_counts(kmf0, kmf1, ax=ax, rows_to_show=['At risk'])

    plt.tight_layout()
    plt.savefig("./", dpi=300, bbox_inches='tight')
    plt.show()


    # 控制台输出 Log-rank 结果
    print(f"Log-rank P值: {p_value:.4f}")

    # Cox 回归分析
    cph = CoxPHFitter()
    cph.fit(df[["time", "status", "MVI"]], duration_col="time", event_col="status")
    cph.print_summary()


# 分析数据集
analyze_dataset(train_df, "Training Cohort")
# analyze_dataset(internal_df, "Internal Test Cohort")
# analyze_dataset(external_df, "External Test Cohort")
