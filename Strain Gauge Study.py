#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:57:10 2025

@author: lucymccoy
"""
import pandas as pd
import os
import glob  
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


#%% Plot all data
selected_date = "2022-11-05"  # Change this to the specific date of interest!!!
# Convert selected_date string to a datetime object
selected_date_dt = datetime.strptime(selected_date, "%Y-%m-%d")

# Generate time ranges: 5-minute window at every hour from 00:05 to 23:05
time_ranges = [(selected_date_dt.replace(hour=hour, minute=5), 
                selected_date_dt.replace(hour=hour, minute=10)) 
               for hour in range(1,13)]

# Directory to save plots
base_dir = "/Users/lucymccoy/Desktop/thesis/coding/FINAL THESIS GRAPHICS"

# First folder: named for the selected_date
date_directory = os.path.join(base_dir, f"{selected_date}_plot_library")
os.makedirs(date_directory, exist_ok=True)

# Define file paths for sensors
base_path = "/Users/lucymccoy/Desktop/thesis/coding/MX3D/"

# Use glob to find the correct sensor files for the selected date
strain_file = glob.glob(os.path.join(base_path, selected_date, "SG_*.npy.pkl.gz"))[0]
temp_file = glob.glob(os.path.join(base_path, selected_date, "T_*.npy.pkl.gz"))[0]

strain_columns = [
    'Timstamp', # 0
    'MX1615-B-R:01-Channel01-StrainGauge-SG01', # 1
    'MX1615-B-R:01-Channel02-StrainGauge-SG02', # 2
    'MX1615-B-R:01-Channel03-StrainGauge-SG03', # 3
    'MX1615-B-R:01-Channel04-StrainGauge-SG04', # ¢ 
    'MX1615-B-R:01-Channel05-StrainGauge-SG05', # 5
    'MX1615-B-R:01-Channel06-StrainGauge-SG06', # 6
    'MX1615-B-R:01-Channel07-StrainGauge-SG07', # 7
    'MX1615-B-R:01-Channel08-StrainGauge-SG08', # 8
    'MX1615-B-R:01-Channel09-StrainGauge-SG09', # 9
    'MX1615-B-R:01-Channel15-StrainGauge-SG10', # 10
    'MX1615-B-R:01-Channel10-StrainGauge-SG11', # 11
    'MX1615-B-R:02-Channel02-StrainGauge-SG12', # 12
    'MX1615-B-R:02-Channel03-StrainGauge-SG13', # 13
    'MX1615-B-R:02-Channel04-StrainGauge-SG14', # 14
    'MX1615-B-R:02-Channel05-StrainGauge-SG15', # 15
    'MX1615-B-R:02-Channel06-StrainGauge-SG16', # 16
    'MX1615-B-R:02-Channel07-StrainGauge-SG17', # 17
    'MX1615-B-R:02-Channel08-StrainGauge-SG18', # 18
    'MX1615-B-R:02-Channel09-StrainGauge-SG19', # 19
    'MX1615-B-R:02-Channel10-StrainGauge-SG20', # 20
    'MX1615-B-R:02-Channel11-StrainGauge-SG21', # 21
    'MX1615-B-R:03-Channel01-StrainGauge-SG22', # 22
    'MX1615-B-R:02-Channel12-StrainGauge-SG23', # 23 <-- INTEREST
    'MX1615-B-R:03-Channel02-StrainGauge-SG24', # 24
    'MX1615-B-R:03-Channel03-StrainGauge-SG25', # 25
    'MX1615-B-R:03-Channel04-StrainGauge-SG26', # 26
    'MX1615-B-R:03-Channel05-StrainGauge-SG27', # 27
    'MX1615-B-R:04-Channel01-StrainGauge-SG28', # 28
    'MX1615-B-R:04-Channel02-StrainGauge-SG29', # 29
    'MX1615-B-R:04-Channel03-StrainGauge-SG30', # 30
    'MX1615-B-R:04-Channel04-StrainGauge-SG31', # 31
    'MX1615-B-R:03-Channel06-StrainGauge-SG32', # 32
    'MX1615-B-R:03-Channel07-StrainGauge-SG33', # 33
    'MX1615-B-R:03-Channel08-StrainGauge-SG34', # 34
    'MX1615-B-R:03-Channel09-StrainGauge-SG35', # 35
    'MX1615-B-R:03-Channel10-StrainGauge-SG36', # 36
    'MX1615-B-R:03-Channel11-StrainGauge-SG37', # 37
    'MX1615-B-R:03-Channel12-StrainGauge-SG38', # 38
    'MX1615-B-R:03-Channel13-StrainGauge-SG39', # 39
    'MX1615-B-R:04-Channel05-StrainGauge-SG40' # 40
]

temp_columns = [
    'Timestamp', # 0
    'MX1615-B-R:01-Channel11-Thermistor-T01', # 1
    'MX1615-B-R:01-Channel12-Thermistor-T02', # 2 <-- INTEREST
    'MX1615-B-R:02-Channel13-Thermistor-T03', # 3 <-- INTEREST
    'MX1615-B-R:02-Channel14-Thermistor-T04', # 4
    'MX1615-B-R:03-Channel14-Thermistor-T05', # 5
    'MX1615-B-R:02-Channel15-Thermistor-T06', # 6
    'MX1615-B-R:02-Channel16-Thermistor-T07', # 7 <-- INTEREST
    'MX1615-B-R:04-Channel06-Thermistor-T08', # 8
    'MX1615-B-R:03-Channel15-Thermistor-T09', # 9 <-- INTEREST
    'MX1615-B-R:03-Channel16-Thermistor-T10', # 10
    'MX1615-B-R:04-Channel07-Thermistor-T11', # 11
    'MX1615-B-R:04-Channel08-Thermistor-T12', # 12
    'MX1615-B-R:04-Channel09-Thermistor-T13', # 13
    'MX1615-B-R:04-Channel10-Thermistor-T14', # 14 
    'MX1615-B-R:04-Channel11-Thermistor-T15', # 15 <-- INTEREST
    'MX1615-B-R:04-Channel12-Thermistor-T16' # 16 <-- INTEREST
]

# Function to load multiple sensor data (temp)
def load_sensor_data_multi(file_path, column_indices):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    with gzip.open(file_path, "rb") as f:
        np_data = pickle.load(f)
    timestamps = np_data[:, 0]
    sensor_data = np_data[:, column_indices]
    return timestamps, sensor_data

# Thermistor indices for T02, T03, T07, T09, T15, T16 (from temp_columns list)
selected_temp_indices = [2, 3, 7, 9, 15, 16]

selected_temp_indices_name = [temp_columns[i] for i in selected_temp_indices]

# Load multi-channel temp data
temp_timestamps, temp_data_multi = load_sensor_data_multi(temp_file, selected_temp_indices)

# Compute mean temperature across selected channels
mean_temp_data = np.mean(temp_data_multi, axis=1)

# Convert timestamp
temp_timestamps_datetime = pd.to_datetime(temp_timestamps)

plt.figure(figsize=(12, 6))
plt.plot(temp_timestamps_datetime, mean_temp_data, label="Mean Temperature", color='orange')
# Format the thermistor list for better readability
included_thermistors = '\n'.join(selected_temp_indices_name)
# Add annotation to **upper-right** of the plot
plt.annotate(
    f"Included Thermistors:\n{included_thermistors}",
    xy=(0.02, 0.94), xycoords='axes fraction',  # Upper-left inside the plot
    fontsize=10,
    ha='left', va='top',
    multialignment='left',  # Ensures text inside the box is left-aligned
    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", lw=1)
)
plt.xlabel("Timestamp")
plt.ylabel("Mean Temperature (°C)")
plt.title(f"Mean Thermistor Temperature on {selected_date}", fontsize=14, fontweight='bold', y=1.02)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# Save plot
#plt.savefig(os.path.join(plot_directory, f"{selected_date}_Mean_Temperature_Selected_Thermistors.png"), dpi=300)
# Show plot
plt.show()

# -Select Strain Channels
# You can manually define or use input() for CLI
selected_channel_indices = list(range(1, 41))  # 1 to 40 inclusive

selected_channel_names = [strain_columns[i] for i in selected_channel_indices]

# Laod data
with gzip.open(strain_file, 'rb') as f:
    strain_data_array = pickle.load(f)

strain_df = pd.DataFrame(strain_data_array, columns=strain_columns)

# Convert timestamps
strain_timestamps = strain_data_array[:, 0]
strain_timestamps_datetime = pd.to_datetime(strain_timestamps)

# Create plot
plt.figure(figsize=(14, 7))

for idx in selected_channel_indices:
    channel_name = strain_columns[idx]
    sensor_data = strain_data_array[:, idx]
    plt.plot(strain_timestamps_datetime, sensor_data, label=channel_name)

plt.xlabel("Timestamp")
plt.ylabel("Strain (N)")
plt.suptitle(
    f"Strain Gauge Data for All Channels\n{selected_date}",
    fontsize=16,
    fontweight='bold',
    y=1.02
)
plt.legend(
    loc="center left",       # Moves it to the left of plot area
    bbox_to_anchor=(1.05, 0.5), # Pushes legend outside the plot
    fontsize=10,
    ncol=2,
    title="Strain Channels"
)
plt.grid(True)
plt.xticks(rotation=45)

# Generate a short-safe filename with SG numbers only
short_channel_ids = "_".join([strain_columns[i][-4:] for i in selected_channel_indices])
save_name = f"{selected_date}_Strain_Gauge_Channels_Data_(NOT_SG03).png"
save_path = os.path.join(date_directory, save_name)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

#%% Plot Cleaned Data
selected_date = "2022-11-05"  # Change this to the specific date of interest!!!
selected_date_dt = datetime.strptime(selected_date, "%Y-%m-%d")

# Directory to save plots
base_dir = "/Users/lucymccoy/Desktop/thesis/coding/FINAL THESIS GRAPHICS"
date_directory = os.path.join(base_dir, f"{selected_date}_plot_library")
os.makedirs(date_directory, exist_ok=True)

# Define file path
base_path = "/Users/lucymccoy/Desktop/thesis/coding/MX3D/"
strain_file = glob.glob(os.path.join(base_path, selected_date, "SG_*.npy.pkl.gz"))[0]

strain_columns = [
    'Timstamp',  # 0
    'MX1615-B-R:01-Channel01-StrainGauge-SG01', # 1
    'MX1615-B-R:01-Channel02-StrainGauge-SG02',
    'MX1615-B-R:01-Channel03-StrainGauge-SG03',
    'MX1615-B-R:01-Channel04-StrainGauge-SG04',
    'MX1615-B-R:01-Channel05-StrainGauge-SG05',
    'MX1615-B-R:01-Channel06-StrainGauge-SG06',
    'MX1615-B-R:01-Channel07-StrainGauge-SG07',
    'MX1615-B-R:01-Channel08-StrainGauge-SG08',
    'MX1615-B-R:01-Channel09-StrainGauge-SG09',
    'MX1615-B-R:01-Channel15-StrainGauge-SG10',
    'MX1615-B-R:01-Channel10-StrainGauge-SG11',
    'MX1615-B-R:02-Channel02-StrainGauge-SG12',
    'MX1615-B-R:02-Channel03-StrainGauge-SG13',
    'MX1615-B-R:02-Channel04-StrainGauge-SG14',
    'MX1615-B-R:02-Channel05-StrainGauge-SG15',
    'MX1615-B-R:02-Channel06-StrainGauge-SG16',
    'MX1615-B-R:02-Channel07-StrainGauge-SG17',
    'MX1615-B-R:02-Channel08-StrainGauge-SG18',
    'MX1615-B-R:02-Channel09-StrainGauge-SG19',
    'MX1615-B-R:02-Channel10-StrainGauge-SG20',
    'MX1615-B-R:02-Channel11-StrainGauge-SG21',
    'MX1615-B-R:03-Channel01-StrainGauge-SG22',
    'MX1615-B-R:02-Channel12-StrainGauge-SG23',
    'MX1615-B-R:03-Channel02-StrainGauge-SG24',
    'MX1615-B-R:03-Channel03-StrainGauge-SG25',
    'MX1615-B-R:03-Channel04-StrainGauge-SG26',
    'MX1615-B-R:03-Channel05-StrainGauge-SG27',
    'MX1615-B-R:04-Channel01-StrainGauge-SG28',
    'MX1615-B-R:04-Channel02-StrainGauge-SG29',
    'MX1615-B-R:04-Channel03-StrainGauge-SG30',
    'MX1615-B-R:04-Channel04-StrainGauge-SG31',
    'MX1615-B-R:03-Channel06-StrainGauge-SG32',
    'MX1615-B-R:03-Channel07-StrainGauge-SG33',
    'MX1615-B-R:03-Channel08-StrainGauge-SG34',
    'MX1615-B-R:03-Channel09-StrainGauge-SG35',
    'MX1615-B-R:03-Channel10-StrainGauge-SG36',
    'MX1615-B-R:03-Channel11-StrainGauge-SG37',
    'MX1615-B-R:03-Channel12-StrainGauge-SG38',
    'MX1615-B-R:03-Channel13-StrainGauge-SG39',
    'MX1615-B-R:04-Channel05-StrainGauge-SG40'
]

# Select channels 1 to 40
selected_channel_indices = list(range(1, 41))

# Load data
with gzip.open(strain_file, 'rb') as f:
    strain_data_array = pickle.load(f)

strain_timestamps = strain_data_array[:, 0]
strain_timestamps_datetime = pd.to_datetime(strain_timestamps)

# Create Plot
plt.figure(figsize=(14, 7))
plotted_channels = []
excluded_channels = []

for idx in selected_channel_indices:
    channel_name = strain_columns[idx]
    sensor_data = strain_data_array[:, idx]
    
    max_val = np.nanmax(sensor_data)
    min_val = np.nanmin(sensor_data)

    if max_val < 1e15 and min_val > -0.8e15:
        plt.plot(strain_timestamps_datetime, sensor_data, label=channel_name, linewidth=0.8)
        plotted_channels.append(channel_name[-4:])  # e.g. "SG01"
    else:
        excluded_channels.append((channel_name, max_val, min_val))
        print(f"⚠️ Skipping {channel_name} — exceeds ±1e15: max={max_val:.2e}, min={min_val:.2e}")

# Finalize plot
plt.xlabel("Timestamp")
plt.ylabel("Strain (N)")
plt.suptitle(
    f"Strain Gauge Data for All Channels (Excl. Outliers)\n{selected_date}",
    fontsize=16,
    fontweight='bold',
    y=1.02
)

plt.legend(
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    fontsize=10,
    ncol=2,
    title="Strain Channels"
)

plt.grid(True)
plt.xticks(rotation=45)

plt.ylim(0, 2000)

# Save cleaned filename
short_channel_ids = "_".join(plotted_channels)
save_name = f"{selected_date}_Strain_CLEANED_Channels_.png"
save_path = os.path.join(date_directory, save_name)

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {save_path}")
print(f"{len(excluded_channels)} channel(s) excluded due to extreme values:\n")
for chan, max_v, min_v in excluded_channels:
    print(f"  - {chan} | max = {max_v:.2e} | min = {min_v:.2e}")


#%% PLOT SPECIFIC SG CHANNELS

# Define channels
channel_indices = [7, 21]
channel_names = [strain_columns[i] for i in channel_indices]

# Load SG12 + SG14 into DataFrame
def load_sensor_df(file_path, indices, col_names):
    with gzip.open(file_path, "rb") as f:
        np_data = pickle.load(f)
    df = pd.DataFrame(np_data[:, indices], columns=col_names)
    df['Timestamp'] = pd.to_datetime(np_data[:, 0])
    df.set_index('Timestamp', inplace=True)
    return df

# Load strain data
df_strain = load_sensor_df(strain_file, channel_indices, channel_names)

# Plot 17 separate time-windowed figures
for i, (start, end) in enumerate(time_ranges):
    window_df = df_strain.loc[start:end]

    plt.figure(figsize=(10, 4))
    plt.plot(window_df.index, window_df.iloc[:, 0], label='SG21', color='tab:blue')
    plt.plot(window_df.index, window_df.iloc[:, 1], label='SG23', color='tab:red')
    plt.title(f"Strain Gauge 07 & Strain Gauge 21\n{start.strftime('%H:%M')} - {end.strftime('%H:%M')} on {selected_date}", fontsize=12, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Strain (N)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(date_directory, f"{selected_date}_{channel_names}_Strain_Gauge_Data_{start.strftime('%H%M')}_{end.strftime('%H%M')}.png"), dpi=300)  # Save image
    plt.show()
    
#%% SG Channels vs Mean Temp
# Select your strain channels and thermistors
selected_sg_indices = [7,]  # SG21, SG26
selected_temp_indices = [2,]  # T02, T03, T07, ...

# Load temperature data
temp_timestamps, temp_data_multi = load_sensor_data_multi(temp_file, selected_temp_indices)
temp_timestamps = pd.to_datetime(temp_timestamps)

# Load strain data
strain_timestamps, strain_data_multi = load_sensor_data_multi(strain_file, selected_sg_indices)
strain_timestamps = pd.to_datetime(strain_timestamps)

# Result containers
strain_vs_temp = {i: {"mean_temps": [], "mean_strains": [], "labels": []} for i in selected_sg_indices}

# Time windows
for start, end in time_ranges:
    temp_mask = (temp_timestamps >= start) & (temp_timestamps < end)
    temp_filtered = temp_data_multi[temp_mask]
    if temp_filtered.size == 0:
        continue
    mean_temp = np.mean(temp_filtered)

    strain_mask = (strain_timestamps >= start) & (strain_timestamps < end)
    strain_filtered = strain_data_multi[strain_mask]
    if strain_filtered.shape[0] == 0:
        continue

    for idx, sg_index in enumerate(selected_sg_indices):
        sg_data = strain_filtered[:, idx]
        mean_strain = np.mean(sg_data)

        strain_vs_temp[sg_index]["mean_temps"].append(mean_temp)
        strain_vs_temp[sg_index]["mean_strains"].append(mean_strain)
        strain_vs_temp[sg_index]["labels"].append(f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}")

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

for sg_index in selected_sg_indices:
    mean_temps = strain_vs_temp[sg_index]["mean_temps"]
    mean_strains = strain_vs_temp[sg_index]["mean_strains"]
    labels = strain_vs_temp[sg_index]["labels"]
    sg_name = strain_columns[sg_index]

    plt.figure(figsize=(10, 6))
    plt.scatter(mean_temps, mean_strains, color="tab:blue", s=100, edgecolors="k", label="Data")

    # Regression Fit
    X = np.array(mean_temps).reshape(-1, 1)
    y = np.array(mean_strains)
    model = LinearRegression().fit(X, y)
    x_fit = np.linspace(min(mean_temps), max(mean_temps), 100).reshape(-1, 1)
    y_fit = model.predict(x_fit)

    plt.plot(x_fit, y_fit, color="red", linestyle="--",
             label=f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (mean_temps[i], mean_strains[i]), textcoords="offset points", xytext=(5,5), fontsize=10)

    plt.title(f"{sg_name} - Strain vs Mean Temperature\n{selected_date}", fontsize=14, fontweight="bold")
    plt.xlabel("Mean Temperature (°C)")
    plt.ylabel("Strain (N)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save
    save_name = f"{selected_date}_{sg_name}_Strain_vs_Mean_Temp.png"
    save_path = os.path.join(date_directory, save_name)
    plt.savefig(save_path, dpi=300)
    plt.show()
