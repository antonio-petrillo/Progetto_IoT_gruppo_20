"""
Progetto IoT - anno 2025/2026 - Gruppo 20.

Antonio Petrillo - N9700496 - antonio.petrillo4@studenti.unina.it
Alessandro Petrella - -
"""

# builtin libraries
import inspect
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore") # matplotlib complains when runned in headless mode
from pathlib import Path
from functools import reduce

# math related libraries
import numpy as np
import pandas as pd

# machine learning and data science library
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA # for dimensions reduction
from sklearn.inspection import DecisionBoundaryDisplay

# plot libraries,  'use("SVG")' for high quality vector graphics
import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt

# colored printing
from rich import print

# TODO: load from env or argument
DATASET_PATH = Path("./eSports_Sensors_Dataset-master") / "matches"
OUTPUT_PATH = Path("./out")

RANDOM_STATE = 0

IF_PARAMS = {
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

SENSORS_FILES = [
    "eeg_band_power.csv",
    "eeg_metrics.csv",
    "emg.csv",
    "eye_tracker.csv",
    "facial_skin_temperature.csv",
    "gsr.csv",
    "heart_rate.csv",
    "imu_chair_back.csv",
    "imu_chair_seat.csv",
    "imu_head.csv",
    "imu_left_hand.csv",
    "imu_right_hand.csv",
    "keyboard.csv",
    "mouse.csv",
    "spo2.csv",
]

# log levels
INFO="INFO"
ERROR="ERROR"
WARN="WARN"
LOG="LOG"

def level_to_color(level):
    match level.upper():
        case "ERROR":
            color = "red"
        case "WARN":
            color = "magenta"
        case "INFO":
            color = "green"
        case "LOG":
            color = "blue"
        case _:
            color = "magenta"
    return color

def trace(message, level=INFO):
    function_name = inspect.currentframe().f_back.f_code.co_name
    color = level_to_color(level)

    print(f"[bold {color}][{level} in fn '{function_name}']:[/bold {color}] "
          f"{message}")

def load_sensor(path):
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        df = df.set_index("time").sort_index() # maybe it is not needed
        df = df.apply(pd.to_numeric, errors="coerce") # unparsable val -> NaN (maybe not needed, dataset is very clean)
        df = df.dropna(how="all") # drop Not Allowed
        return df if not df.empty else None
    except Exception as e:
        trace(f"Could not load {path.name}: e", level=ERROR)
        return None

def run_isolation_forest(df):
    clean = df.fillna(df.median(numeric_only=True))
    model = IsolationForest(**IF_PARAMS)
    labels = model.fit_predict(clean)
    scores = model.decision_function(clean)
    return model, labels, scores


COMMON_ARGS_HANDLES = {"c": "#d73027", "s": 30, "edgecolor": "k", "lw": 0.3}
def save_plot_2d(df, model, labels, title, path):
    clean = df.fillna(df.median(numeric_only=True)).values
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(clean)

    var_explained = pca.explained_variance_ratio_ * 100
    model_2d = IsolationForest(**IF_PARAMS)
    model_2d.fit(X_2d)

    is_out = labels == -1
    colors = np.where(is_out, -1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{title}\nOutliers: {is_out.sum()} / {len(labels)} ({100 * is_out.mean():.1f}%)"
        f" | PCA variance explained: PC1={var_explained[0]:.1f}% PC2={var_explained[1]:.1f}%",
        fontsize=11, fontweight="bold",
    )

    disp1 = DecisionBoundaryDisplay.from_estimator(
        model_2d, X_2d,
        response_method="predict",
        alpha=0.4,
        ax=axes[0],
        cmap="RdYlGn",
    )

    scatter1 = disp1.ax_.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=colors, cmap="RdYlGn",
        s=18, edgecolor="k", linewidth=0.3, zorder=5,
    )

    legend_handles = [
        # plt.scatter([], [], c="#d73027", s=30, edgecolor="k", lw=0.3, label="Outlier"),
        plt.scatter([], [], label="Outlier", **common_args_handles),
        plt.scatter([], [], label="Inlier", **common_args_handles),
    ]

    axes[0].legend(handles=legend_handles, title="IF label", fontsize=8)
    axes[0].grid(True, alpha=0.2)

    disp2 = DecisionBoundaryDisplay.from_estimator(
        model_2d, X_2d,
        response_method="decision_function",
        alpha=0.5,
        ax=axes[1],
        cmap="RdYlGn",
    )
    scatter2 = disp2.ax_.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=colors, cmap="RdYlGn",
        s=18, edgecolor="k", linewidth=0.3, zorder=5,
    )
    plt.colorbar(disp2.ax_.collections[1], ax=axes[1], label="Anomaly Score\n(lower = more anomalous)")
    axes[1].set_title("Path-length decision boundary\n(anomaly score)")
    axes[1].set_xlabel(f"PC1 ({var_explained[0]:.1f})% var")
    axes[1].set_ylabel(f"PC2 ({var_explained[1]:.1f})% var")
    axes[1].legend(handles=legend_handles, title="IF label", fontsize=8)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", format="svg")
    plt.close(fig)

def save_plot(df, labels, scores, title, path):
    cols     = df.columns.tolist()
    n_rows   = len(cols) + 1
    time_idx = df.index.values
    is_out   = labels == -1

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, max(3, 2.2 * n_rows)),
        sharex=False,
        gridspec_kw={"hspace": 0.55},
    )
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        f"{title}\nOutliers: {is_out.sum()} / {len(labels)} ({100*is_out.mean():.1f}%)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for i, col in enumerate(cols):
        ax  = axes[i]
        val = df[col].values
        ax.plot(time_idx, val, color="#4C72B0", linewidth=0.7, alpha=0.85)
        ax.scatter(time_idx[is_out], val[is_out], color="red", s=15, zorder=5)
        ax.set_ylabel(col, fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    ax_s = axes[-1]
    ax_s.plot(time_idx, scores, color="#444444", linewidth=0.7)
    ax_s.axhline(0, color="orange", linewidth=1, linestyle="--")
    ax_s.fill_between(time_idx, scores, 0,
                      where=(scores < 0), color="red", alpha=0.15)
    ax_s.set_ylabel("Anomaly\nScore", fontsize=8)
    ax_s.set_xlabel("Time (s)", fontsize=9)
    ax_s.tick_params(labelsize=7)
    ax_s.grid(True, alpha=0.25)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", format="svg")
    plt.close(fig)

KEY_COLUMN = "time"
def process_player(match_id, player_id):
    in_path = DATASET_PATH / match_id / player_id

    dfs = []
    for sensor in SENSORS_FILES:
        path = in_path / sensor

        if not path.exists():
            trace(f"sensor not found: {path}", level=WARN)
            continue

        df = load_sensor(path)
        model, labels, scores = run_isolation_forest(df)
        basename = sensor.replace(".csv", "")
        out_path = OUTPUT_PATH / match_id / player_id / f"{basename}.svg"
        out_path_2d = OUTPUT_PATH / match_id / player_id / f"{basename}_2d.svg"
        save_plot(df, labels, scores, f"{match_id}_{player_id}_{sensor}", out_path)
        if df.shape[1] >= 2:
            save_plot_2d(df, model, labels, f"{match_id}_{player_id}_{sensor}_2d", out_path_2d)

        cols = {col: f"{col}_{sensor}" for col in df.columns if col != KEY_COLUMN}
        dfs.append(df.rename(columns = cols))

    combined = reduce(lambda l, r: l.join(r, how="inner"), dfs)
    model, labels, scores = run_isolation_forest(combined)
    out_path = OUTPUT_PATH / match_id / player_id / "sensors_combined.svg"
    save_plot_2d(combined, model, labels, f"{match_id}_{player_id}_sensors_combined", out_path)
    return combined

def pipeline():
    report = {
        "least_anomalous_match": None,
        "most_anomalous_match": None,
    }
    for match_id in DATASET_PATH.iterdir():
        if not match_id.exists():
            continue

        dfs = []
        for player_id in match_id.iterdir():
            if not player_id.exists():
                trace("player not found", level=ERROR)
                sys.exit(1)
            if player_id.name.endswith(".json") or player_id.name.endswith(".csv"):
                continue

            df = process_player(match_id.name, player_id.name)
            cols = {col: f"{col.replace('.csv', '')}_p{player_id.name[-1]}" for col in df.columns if col != KEY_COLUMN}
            dfs.append(df.rename(columns = cols))

        combined = reduce(lambda l, r: l.join(r, how="inner"), dfs)
        model, labels, scores = run_isolation_forest(combined)
        out_path = OUTPUT_PATH / match_id.name / "all_players.svg"
        save_plot_2d(combined, model, labels, f"{match_id.name}_all_players_combined", out_path)

        mean = scores.mean()
        if report["least_anomalous_match"] is None or report["least_anomalous_match"]["score"] > mean:
            report["least_anomalous_match"] = {"id": match_id.name, "score": mean}

        if report["most_anomalous_match"] is None or report["most_anomalous_match"]["score"] < mean:
            report["most_anomalous_match"] = {"id": match_id.name, "score": mean}


    report_path = OUTPUT_PATH / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f)


def main():
    pipeline()

if __name__ == "__main__":
    main()
