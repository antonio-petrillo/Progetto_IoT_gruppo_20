import argparse
import warnings
from pathlib import Path
import os
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

#################################################################
########################### DEBUG ###############################
#################################################################

TRACE = True

#################################################################

#################################################################
####################### GLOBAL CONTEXT ##########################
#################################################################

load_dotenv()
warnings.filterwarnings("ignore")

DATASET_PATH = Path(os.getenv("DATASET_PATH"))
TIME_COL = "time"

SENSOR_CONFIG = {
    "emg": {
        "value_cols": ["emg_right_hand", "emg_left_hand"],
        "agg":        ["mean", "std", "max"],
    },
    "eye_tracker": {
        "value_cols": ["gaze_movement", "pupil_diameter"],
        "agg":        ["mean", "std"],
    },
    "facial_skin_temperature": {
        "value_cols": ["facial_skin_temperature"],
        "agg":        ["mean", "std", "min", "max"],
    },
    "gsr": {
        "value_cols": ["gsr"],
        "agg":        ["mean", "std", "min", "max"],
    },
    "imu_chair_back": {
        "value_cols": ["linaccel_x", "linaccel_y", "linaccel_z",
                       "gyro_x", "gyro_y", "gyro_z",
                       "euler_x", "euler_y", "euler_z"],
        "agg":        ["mean", "std"],
    },
    "imu_chair_seat": {
        "value_cols": ["linaccel_x", "linaccel_y", "linaccel_z",
                       "gyro_x", "gyro_y", "gyro_z",
                       "euler_x", "euler_y", "euler_z"],
        "agg":        ["mean", "std"],
    },
    "imu_left_hand": {
        "value_cols": ["linaccel_x", "linaccel_y", "linaccel_z",
                       "gyro_x", "gyro_y", "gyro_z",
                       "euler_x", "euler_y", "euler_z"],
        "agg":        ["mean", "std"],
    },
    "imu_right_hand": {
        "value_cols": ["linaccel_x", "linaccel_y", "linaccel_z",
                       "gyro_x", "gyro_y", "gyro_z",
                       "euler_x", "euler_y", "euler_z"],
        "agg":        ["mean", "std"],
    },
    "keyboard": {
        "value_cols": ["buttons_pressed"],
        "agg":        ["sum", "mean", "std"],
    },
    "mouse": {
        "value_cols": ["mouse_movement", "mouse_clicks"],
        "agg":        ["sum", "mean", "std"],
    },
    "spo2": {
        "value_cols": ["spo2"],
        "agg":        ["mean", "std", "min"],
    },
}

SENSOR_GROUPS = {
    "muscolare (EMG)":          ["emg"],
    "oculare":                  ["eye_tracker"],
    "fisiologico (GSR+SpO2+T)": ["gsr", "spo2", "facial_skin_temperature"],
    "motorio (mouse+keyboard)": ["mouse", "keyboard"],
    "posturale (IMU sedia)":    ["imu_chair_back", "imu_chair_seat"],
    "tutti i sensori":          list(SENSOR_CONFIG.keys()),
}

#################################################################

#################################################################
######################### LOAD DATA  ############################
#################################################################


def sessions_from_path(path):
    sessions = []
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} not found")

    for _match in path.iterdir():
        if not _match.is_dir():
            continue

        print(_match.name)

        for player in _match.iterdir():
            if not player.is_dir():
                continue
            paths = {}
            for s in SENSOR_CONFIG:
                val = player / f"{s}.csv"
                if val.exists():
                    paths[s] = val
                elif TRACE:
                    print(f"{val} does not exist.")

            if paths:
                sessions.append({
                    "match_id": _match.name,
                    "player_id": player.name,
                    "paths": paths,
                })
    return sessions

def load_sensor_for_player(player, sensor):
    """Load the specific 'sensor' for 'player' into a pandas Dataframe."""
    path = player["paths"][sensor]
    if path is None:
        if TRACE:
            print(f"Player {player} does not have data for sensor {sensor}.")
        return None

    try:
        return pd.read_csv(path)
    except Exception as e:
        if TRACE:
            print(f"Error in 'load_sensor_for_player'"
                  f"Params: ({player}, {sensor})"
                  f"Exception: {e}")
        return None

def build_player_matrix(player):
    """doc here..."""

    dataframes = []

    for sensor, config in SENSOR_CONFIG.items():
        df = load_sensor_for_player(player, sensor)
        if df is None:
            if TRACE:
                print("Unexpected in 'build_player_matrix'")
            continue

        available_cols = [c for c in config["value_cols"] if c in df.columns]
        if not available_cols:
            if TRACE:
                print(f"Unexpected in 'build_player_matrix', dataframe has no columns"
                      f"({player}, {sensor})")
            continue

        sub = df[[TIME_COL] + available_cols].copy()
        sub[TIME_COL] = sub[TIME_COL].round().astype(int)
        sub = sub.groupby(TIME_COL)[available_cols].mean()
        sub.columns = [f"{sensor}__{c}" for c in available_cols]
        dataframes.append(sub)

    if not dataframes:
        if TRACE:
                print(f"Unexpected in 'build_player_matrix', player has no data"
                      f"({player}, {sensor})")
        return None

    combined = pd.concat(dataframes, axis=1).sort_index()
    return combined.interpolate(method="linear", limit_direction="both").dropna(axis=1)



#################################################################

#################################################################
#################### CHECK FOR ANOMALIES ########################
#################################################################

def anomalies_player_single_attribute(player):
    results = {}

    for sensor, config in SENSOR_CONFIG:
        df = load_sensor_for_player(player, sensor)
        if df is None:
            if TRACE:
                print("Unexpected in 'anomalies_player_single_attribute'")
            continue

        for col in config["value_cols"]:
            if col not in df.columns:
                continue

            sub = df[[TIME_COL, COL]].dropna()
            if len(sub) < 10:
                continue

            X = StandardScaler().fit_transform(sub[[col]].values)
            clf = IsolationForest(contamination=0.05, random_state=0) # use same random state for now
            preds = clf.fit_predict(X)

            anomaly_mask = pd.Series(
                preds == -1,
                index = sub[TIME_COL].values,
                name=f"{sensor}__{col}",
            )
            results[f"{sensor}__{col}"] = anomaly_mask

            if TRACE:
                print(f"    [single_attr] player={player['player_id']}"
                      f"{sensor}/{col}: {anomaly_mask.sum()}/{len(anomaly_mask)} anomalies")

    return results


def anomalies_player_all_attributes(player):
    """Use isolation forest on player with all the attributes at the same time"""
    pass

def anomalies_match(match_):
    """Use isolation forest on all player in the same match at the same time"""
    pass

#################################################################

def main():
    sessions = sessions_from_path(DATASET_PATH)
    print(sessions)

if __name__ == "__main__":
    main()
