from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


REPO_ROOT = Path(__file__).resolve().parent
PREDICTION_CSV = REPO_ROOT / "result" / "train_data" / "timesfm_price_predictions.csv"
SAVE_DIR = REPO_ROOT / "result" / "pic"

WINDOW_INDEX = 4
NUM_WINDOWS = 1
POINTS_PER_DAY = 96
X_LABEL_STEP = 2


def calculate_tcr(true_arr, pred_arr):
    true_arr = np.asarray(true_arr, dtype=float)
    pred_arr = np.asarray(pred_arr, dtype=float)

    true_diff = true_arr[1:] - true_arr[:-1]
    pred_diff = pred_arr[1:] - pred_arr[:-1]

    true_dir = np.sign(true_diff)
    pred_dir = np.sign(pred_diff)

    up_mask = true_dir > 0
    down_mask = true_dir < 0

    up_count = np.sum(up_mask)
    down_count = np.sum(down_mask)

    correct_up = np.sum(up_mask & (pred_dir > 0))
    correct_down = np.sum(down_mask & (pred_dir < 0))

    tcr_up = correct_up / up_count if up_count > 0 else np.nan
    tcr_down = correct_down / down_count if down_count > 0 else np.nan

    if np.isnan(tcr_up) and np.isnan(tcr_down):
        tcr = np.nan
    elif np.isnan(tcr_up):
        tcr = tcr_down
    elif np.isnan(tcr_down):
        tcr = tcr_up
    else:
        tcr = (tcr_up + tcr_down) / 2

    return tcr_up * 100, tcr_down * 100, tcr * 100


def load_prediction_data():
    df = pd.read_csv(PREDICTION_CSV, encoding="utf-8-sig")
    required_columns = ["预测窗口", "预测起始日", "预测点", "时间", "真实电价", "预测电价", "误差"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"预测结果缺少列: {', '.join(missing)}")

    df["时间"] = pd.to_datetime(df["时间"])
    df = df.sort_values(["预测窗口", "预测点"]).reset_index(drop=True)
    return df


def select_windows(df):
    end_window = WINDOW_INDEX + NUM_WINDOWS
    selected = df[(df["预测窗口"] >= WINDOW_INDEX) & (df["预测窗口"] < end_window)].copy()
    if selected.empty:
        raise ValueError(f"没有找到预测窗口: {WINDOW_INDEX} - {end_window - 1}")
    return selected.reset_index(drop=True)


def calculate_metrics(true_plot, pred_plot):
    error = pred_plot - true_plot
    abs_error = np.abs(error)

    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(error ** 2))
    bias = np.mean(error)
    max_abs_error = np.max(abs_error)

    eps = 1e-8
    bias_rate = np.divide(
        abs_error,
        np.abs(true_plot),
        out=np.full_like(abs_error, np.nan, dtype=float),
        where=np.abs(true_plot) > eps,
    )
    bias_rate_percent = bias_rate * 100
    mape = np.nanmean(bias_rate_percent)

    if np.std(pred_plot) > 0 and np.std(true_plot) > 0:
        corr = np.corrcoef(pred_plot, true_plot)[0, 1]
    else:
        corr = np.nan

    return {
        "error": error,
        "abs_error": abs_error,
        "bias_rate_percent": bias_rate_percent,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "mape": mape,
        "corr": corr,
        "max_abs_error": max_abs_error,
    }


def print_summary(selected, metrics, tcr_up, tcr_down, tcr_total):
    print("预测数据预览:")
    print(selected[["预测窗口", "预测起始日", "预测点", "时间", "真实电价", "预测电价", "误差"]].head(20))
    print(
        f"\nWindow: {WINDOW_INDEX} - {WINDOW_INDEX + NUM_WINDOWS - 1}\n"
        f"Points: {len(selected)}\n"
        f"MAE: {metrics['mae']:.2f}\n"
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"Bias: {metrics['bias']:.2f}\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"Corr: {metrics['corr']:.3f}\n"
        f"Max AE: {metrics['max_abs_error']:.2f}\n"
        f"TCR_up: {tcr_up:.2f}%\n"
        f"TCR_down: {tcr_down:.2f}%\n"
        f"TCR: {tcr_total:.2f}%"
    )


def plot_pred_true(selected, true_plot, pred_plot, tcr_up, tcr_down, tcr_total):
    time_steps = np.arange(len(pred_plot))
    time_labels = selected["时间"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    title_day = selected["预测起始日"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(time_steps, pred_plot, label="预测值", linewidth=3, color="#4F81BD")
    ax.plot(time_steps, true_plot, label="真实值", linewidth=3, color="#C0504D")

    for day_index in range(1, NUM_WINDOWS):
        ax.axvline(x=day_index * POINTS_PER_DAY, linestyle=":", linewidth=1.5, color="gray")

    title_text = (
        f"{title_day}\n"
        f"上涨趋势捕获率 TCR_up  = {tcr_up:.2f}%\n"
        f"下跌趋势捕获率 TCR_down = {tcr_down:.2f}%\n"
        f"综合趋势捕获率 TCR    = {tcr_total:.2f}%"
    )
    ax.set_title(title_text, fontsize=18, fontweight="bold", pad=25)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("电价", fontsize=12)
    ax.set_xticks(time_steps[::X_LABEL_STEP])
    ax.set_xticklabels([time_labels[i] for i in time_steps[::X_LABEL_STEP]], rotation=90, fontsize=10)
    ax.grid(True, axis="y", linestyle="-", alpha=0.4)
    ax.grid(False, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(bottom=0.35, top=0.78)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"pred_true_TCR_window_{WINDOW_INDEX}_{NUM_WINDOWS}days.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"预测值与真实值对比图已保存到: {save_path}")


def plot_bias_rate(metrics):
    bias_rate_percent = metrics["bias_rate_percent"]
    valid_bias_rate = bias_rate_percent[np.isfinite(bias_rate_percent)]
    total_count = len(valid_bias_rate)
    if total_count == 0:
        raise ValueError("无法计算偏差率: 真实电价全为 0 或无有效点")

    ratio_lt_5 = np.sum(valid_bias_rate < 5) / total_count * 100
    ratio_5_10 = np.sum((valid_bias_rate >= 5) & (valid_bias_rate < 10)) / total_count * 100
    ratio_10_15 = np.sum((valid_bias_rate >= 10) & (valid_bias_rate < 15)) / total_count * 100
    ratio_15_20 = np.sum((valid_bias_rate >= 15) & (valid_bias_rate < 20)) / total_count * 100
    ratio_gt_20 = np.sum(valid_bias_rate >= 20) / total_count * 100

    fig, ax = plt.subplots(figsize=(14, 7))
    x_bias = np.arange(1, len(bias_rate_percent) + 1)
    ax.plot(x_bias, bias_rate_percent, label="偏差率(%)", linewidth=3, color="#4F81BD")

    bias_title = (
        "偏差率(%)\n"
        f"小于5%偏差的占比为 {ratio_lt_5:.2f}%，"
        f" 大于5%小于10%的偏差的占比为 {ratio_5_10:.2f}%，"
        f" 大于10%小于15%的偏差的占比为 {ratio_10_15:.2f}%，\n"
        f"大于15%小于20%的偏差的占比为 {ratio_15_20:.2f}%，"
        f" 大于20%的占比为 {ratio_gt_20:.2f}%。"
    )
    ax.set_title(bias_title, fontsize=17, fontweight="bold", pad=25)
    ax.set_xlabel("时间点", fontsize=12)
    ax.set_ylabel("偏差率(%)", fontsize=12)
    ax.set_xticks(x_bias[::X_LABEL_STEP])
    ax.set_xticklabels(x_bias[::X_LABEL_STEP], fontsize=10)
    ax.grid(True, axis="y", linestyle="-", alpha=0.4)
    ax.grid(False, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=12)
    fig.subplots_adjust(bottom=0.22, top=0.72)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"bias_rate_distribution_window_{WINDOW_INDEX}_{NUM_WINDOWS}days.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"偏差率图已保存到: {save_path}")


def main():
    df = load_prediction_data()
    selected = select_windows(df)

    true_plot = selected["真实电价"].to_numpy(dtype=float)
    pred_plot = selected["预测电价"].to_numpy(dtype=float)
    metrics = calculate_metrics(true_plot, pred_plot)
    tcr_up, tcr_down, tcr_total = calculate_tcr(true_plot, pred_plot)

    print_summary(selected, metrics, tcr_up, tcr_down, tcr_total)
    plot_pred_true(selected, true_plot, pred_plot, tcr_up, tcr_down, tcr_total)
    plot_bias_rate(metrics)


if __name__ == "__main__":
    main()
