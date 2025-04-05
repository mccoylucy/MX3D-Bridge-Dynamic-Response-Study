#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:27:54 2025

@author: lucymccoy
"""

import os
import gzip
import pickle
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# General Configurations
selected_date = "2022-04-24"
base_path = "/Users/lucymccoy/Desktop/thesis/coding/MX3D"
plot_save_dir = f"/Users/lucymccoy/Desktop/thesis/coding/FINAL THESIS GRAPHICS/{selected_date}_plot_library/Temp_vs_Min_Load"

# Ensure output directory exists
os.makedirs(plot_save_dir, exist_ok=True)

# File Loacations
temp_file = glob.glob(os.path.join(base_path, selected_date, "T_*.npy.pkl.gz"))[0]
load_file = glob.glob(os.path.join(base_path, selected_date, "LC_*.npy.pkl.gz"))[0]

# Selected Channels
selected_temp_indices = [2, 3, 7, 9, 15, 16]  # T02, T03, T07, T09, T15, T16
load_columns = [
    'Timestamp',
    'MX1615-B-R:02-Channel01-LoadCell-LC01',
    'MX1615-B-R:01-Channel16-LoadCell-LC02',
    'MX1615-B-R:04-Channel15-LoadCell-LC03',
    'MX1615-B-R:04-Channel16-LoadCell-LC04'
]

# Define Time Windows
selected_date_dt = datetime.strptime(selected_date, "%Y-%m-%d")
time_ranges = [(selected_date_dt.replace(hour=hr, minute=5), selected_date_dt.replace(hour=hr, minute=10)) for hr in range(1, 20)]

# Laod Data Functions
def load_sensor_data_multi(file_path, column_indices):
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
    timestamps = pd.to_datetime(data[:, 0])
    sensor_data = data[:, column_indices]
    return timestamps, sensor_data

def load_sensor_column(file_path, column_index):
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
    timestamps = pd.to_datetime(data[:, 0])
    column_data = data[:, column_index]
    return timestamps, column_data

# Load Temp Data
temp_timestamps, temp_data_multi = load_sensor_data_multi(temp_file, selected_temp_indices)

# Loop Load Cells
for lc_index in range(1, 5):
    lc_name = load_columns[lc_index]

    # Load this load cell
    load_timestamps, load_data = load_sensor_column(load_file, lc_index)

    mean_temps = []
    min_loads = []
    time_labels = []

    # Process Time Windows
    for start, end in time_ranges:
        # Temp
        temp_mask = (temp_timestamps >= start) & (temp_timestamps < end)
        temp_filtered = temp_data_multi[temp_mask]
        if temp_filtered.size == 0:
            continue
        mean_temp = np.mean(temp_filtered)

        # Load
        load_mask = (load_timestamps >= start) & (load_timestamps < end)
        load_filtered = load_data[load_mask]
        if load_filtered.size == 0:
            continue
        min_load = np.min(load_filtered)

        # Save values
        mean_temps.append(mean_temp)
        min_loads.append(min_load)
        time_labels.append(f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}")

    # Create Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_temps, min_loads, color='skyblue', edgecolors='k', s=100, label="Data")

    # Linear Fit
    X = np.array(mean_temps).reshape(-1, 1)
    y = np.array(min_loads)
    model = LinearRegression().fit(X, y)
    x_fit = np.linspace(min(mean_temps), max(mean_temps), 100).reshape(-1, 1)
    y_fit = model.predict(x_fit)

    plt.plot(x_fit, y_fit, linestyle="--", color="red", linewidth=2,
             label=f"Fit: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

    # Annotate time ranges
    for i, label in enumerate(time_labels):
        plt.annotate(label, (mean_temps[i], min_loads[i]), textcoords="offset points", xytext=(5, 5), fontsize=10)

    # Labels and formatting
    plt.xlabel("Mean Temperature (°C)")
    plt.ylabel("Minimum Load (kg)")
    plt.title(f"Min Load vs Mean Temp\n{lc_name}\n{selected_date}", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(plot_save_dir, f"{selected_date} {lc_name} Min Load vs Mean Temp"), dpi=300)
    plt.show()
    