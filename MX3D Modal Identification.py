#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:31:50 2025

@author: lucymccoy
"""
#%% Import necessary libraries
import pandas as pd
import os
import glob 
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

#%% Define the selected date (YYYY-MM-DD format)
selected_date = "2022-04-24"  # Change this to the specific date of interest
# Convert selected_date string to a datetime object
selected_date_dt = datetime.strptime(selected_date, "%Y-%m-%d")

# Generate time ranges: 5-minute window at every hour from 00:05 to 23:05
time_ranges = [(selected_date_dt.replace(hour=hour, minute=5), 
                selected_date_dt.replace(hour=hour, minute=10)) 
               for hour in range(1,18)]

# Debugging: Print time ranges
print(f"Investigating data for {selected_date}")
print("Time windows of interest:")
for start, end in time_ranges:
    print(f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}")

#%% Define file paths for sensors
base_path = "/Users/lucymccoy/Desktop/thesis/coding/MX3D"

# Use glob to find the correct sensor files for the selected date
acc_file = glob.glob(os.path.join(base_path, selected_date, "A_*.npy.pkl.gz"))[0]
temp_file = glob.glob(os.path.join(base_path, selected_date, "T_*.npy.pkl.gz"))[0]
load_file = glob.glob(os.path.join(base_path, selected_date, "LC_*.npy.pkl.gz"))[0]

# Define  column headers (assumed identical across days)
acc_columns = [
    'Timestamp',  # 0
    'MX1601-B-R:01-Channel08-Accelerometer-A01X',  # 1
    'MX1601-B-R:01-Channel09-Accelerometer-A01Y',  # 2 <-- INTEREST
    'MX1601-B-R:01-Channel14-Accelerometer-A02X',  # 3
    'MX1601-B-R:01-Channel15-Accelerometer-A02Y',  # 4 <-- INTEREST
    'MX1601-B-R:01-Channel03-Accelerometer-A03X',  # 5
    'MX1601-B-R:01-Channel04-Accelerometer-A03Y',  # 6
    'MX1601-B-R:01-Channel06-Accelerometer-A04X',  # 7
    'MX1601-B-R:01-Channel07-Accelerometer-A04Y',  # 8
    'MX1601-B-R:01-Channel12-Accelerometer-A05X',  # 9
    'MX1601-B-R:01-Channel13-Accelerometer-A05Y',  # 10 <-- INTEREST
    'MX1601-B-R:01-Channel16-Accelerometer-A06X',  # 11
    'MX1601-B-R:01-Channel02-Accelerometer-A06Y',  # 12 <-- INTEREST
    'MX1601-B-R:01-Channel10-Accelerometer-A07X',  # 13
    'MX1601-B-R:01-Channel11-Accelerometer-A07Y',  # 14
    'MX1601-B-R:02-Channel01-Accelerometer-A08X',  # 15
    'MX1601-B-R:02-Channel02-Accelerometer-A08Z',  # 16 
    'MX1601-B-R:02-Channel04-Accelerometer-A09X',  # 17
    'MX1601-B-R:02-Channel05-Accelerometer-A09Z',  # 18
    'MX1601-B-R:02-Channel06-Accelerometer-A10X',  # 19
    'MX1601-B-R:02-Channel07-Accelerometer-A10Z',  # 20 <-- INTEREST
    'MX1601-B-R:02-Channel09-Accelerometer-A11X',  # 21
    'MX1601-B-R:02-Channel10-Accelerometer-A11Z'  # 22 <-- INTEREST
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

load_columns = [
    'Timestamp', # 0
    'MX1615-B-R:02-Channel01-LoadCell-LC01', # 1 <-- INTEREST
    'MX1615-B-R:01-Channel16-LoadCell-LC02', # 2 <-- INTEREST
    'MX1615-B-R:04-Channel15-LoadCell-LC03', # 3 <-- INTEREST
    'MX1615-B-R:04-Channel16-LoadCell-LC04' # 4 <-- INTEREST
]

# Define selected column indices
acc_channel_index = 4
selected_acc_channel_name = acc_columns[acc_channel_index]  

temp_channel_index = 7  
selected_temp_channel_name = temp_columns[temp_channel_index]  

load_channel_index = 1
selected_load_channel_name = load_columns[load_channel_index]  

# Directory to save plots
base_dir = "/Users/lucymccoy/Desktop/thesis/coding/FINAL THESIS GRAPHICS"

# First folder: named for the selected date
date_directory = os.path.join(base_dir, f"{selected_date}_plot_library")
os.makedirs(date_directory, exist_ok=True)

# Second folder: named for the selected_acc_channel_name, inside the date folder
plot_directory = os.path.join(date_directory, f"{selected_acc_channel_name}")
os.makedirs(plot_directory, exist_ok=True)

# Alternative folder for Mean Thermistor window plots
temp_plot_directory = os.path.join(date_directory, "Mean Thermistor Window Plots")
os.makedirs(temp_plot_directory, exist_ok=True)

# Store results for each time window
results = []

#%% Function to load individual sensor data (acc, load, strain)
def load_sensor_data(file_path, column_index):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    with gzip.open(file_path, "rb") as f:
        np_data = pickle.load(f)
    timestamps = np_data[:, 0]  # Extract timestamps
    sensor_data = np_data[:, column_index] # Extract selected sensor data
    return timestamps, sensor_data

#%% Function to load multiple sensor data (temp)
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

#%% Load data for selected day
acc_timestamps, acc_data = load_sensor_data(acc_file, acc_channel_index)
temp_timestamps, temp_data = load_sensor_data(temp_file, temp_channel_index)
load_timestamps, load_data = load_sensor_data(load_file, load_channel_index)

if any(v is None for v in [acc_timestamps, temp_timestamps, load_timestamps]):
    print(f"Missing data for {selected_date}. Exiting.")
    exit()

# Convert timestamps to datetime objects
acc_timestamps_datetime = pd.to_datetime(acc_timestamps)
temp_timestamps_datetime = pd.to_datetime(temp_timestamps)
load_timestamps_datetime = pd.to_datetime(load_timestamps)

#%% Function to normalize each 5-minute window
def normalize_data(acc_data):
    """
    Normalize the accelerometer data by subtracting the mean value for each column.
    """
    means = np.mean(acc_data, axis=0)
    normalized_data = acc_data - means
    return normalized_data

acc_data_normalized = normalize_data(acc_data) 

for start, end in time_ranges:
    acc_mask = (acc_timestamps_datetime >= start) & (acc_timestamps_datetime < end)
    acc_filtered_time = acc_timestamps_datetime[acc_mask]
    acc_filtered_data = acc_data[acc_mask]
    
    if acc_filtered_data.size > 0:
        # Normalize data for each 5-minute window
        acc_filtered_data_normalized = normalize_data(acc_filtered_data)

#%% Plot Selected Sensor Data
plt.figure(figsize=(12, 6))
plt.plot(acc_timestamps_datetime, acc_data_normalized, label=f"{selected_acc_channel_name}")
plt.xlabel("Timestamp")
plt.ylabel("Acceleration (m/s²)")
plt.title(f"Normalised Accelerometer Data\n{selected_date}\n{selected_acc_channel_name}", fontsize=14, fontweight='bold', y=1.02)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_directory, f"{selected_date}_{selected_acc_channel_name}_Normalised_Accelerometer_Data.png"), dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(temp_timestamps_datetime, mean_temp_data, label="Mean Temperature", color='orange')
included_thermistors = '\n'.join(selected_temp_indices_name)
plt.annotate(
    f"Included Thermistors:\n{included_thermistors}",
    xy=(0.02, 0.94), xycoords='axes fraction',  
    fontsize=10,
    ha='left', va='top',
    multialignment='left',  
    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", lw=1)
)
plt.xlabel("Timestamp")
plt.ylabel("Mean Temperature (°C)")
plt.title(f"Mean Thermistor Temperature on {selected_date}", fontsize=14, fontweight='bold', y=1.02)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_directory, f"{selected_date}_Mean_Temperature_Selected_Thermistors.png"), dpi=300)
plt.show()

# (Single Thermistor)
plt.figure(figsize=(12, 6))
plt.plot(temp_timestamps_datetime, temp_data, label=temp_columns[temp_channel_index], color='orange')
plt.xlabel("Timestamp")
plt.ylabel("Temperature (°C)")
plt.title(f"Thermistor Temperature\n{selected_date}\n{selected_temp_channel_name}", fontsize=14, fontweight='bold', y=1.02)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_directory, f"{selected_date}_{temp_columns[temp_channel_index]}_Thermistor_Data.png"), dpi=300)
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(load_timestamps_datetime, load_data, label=f"{selected_load_channel_name}", color='g')
plt.xlabel("Timestamp")
plt.ylabel("Load (kg)")
plt.title(f"Load Cell Data\n{selected_date}\n{selected_load_channel_name}", fontsize=14, fontweight='bold', y=1.02)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plot_directory, f"{selected_date}_{selected_load_channel_name}_Load_Cell_Data.png"), dpi=300)
plt.show()


#%% Plot Load Cell Data for All Channels

# Re-load the entire data array (rather than just the single column)
with gzip.open(load_file, "rb") as f:
    load_data_all = pickle.load(f)  # shape: (N, 5) since index 0 is timestamps + 4 LC channels

# Convert the first column (timestamps)to a pandas datetime for plotting
load_timestamps_all = pd.to_datetime(load_data_all[:, 0])

# Sum across all load-cell channels (columns 1 through end)
total_load = np.sum(load_data_all[:, 1:], axis=1)

plt.figure(figsize=(12, 6))
for i in range(1, len(load_columns)):  
    channel_name = load_columns[i]  
    plt.plot(load_timestamps_all, load_data_all[:, i], label=channel_name)

# Finalize plot
plt.xlabel("Timestamp")
plt.ylabel("Load (kg)")
plt.title(f"All Load Cell Data on {selected_date}", fontsize=14, fontweight='bold', y=1.02)  
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# Save figure
all_channels_plot_path = os.path.join(plot_directory, f"{selected_date}_All_Load_Cells.png")
plt.savefig(os.path.join(plot_directory, f'{selected_date}_All_Load_Cell_Data.png'), format="png", dpi=300)
plt.show()

# Make the plot
plt.figure(figsize=(12, 6))
plt.plot(load_timestamps_all, total_load, label="Total Load")
plt.xlabel("Timestamp")
plt.ylabel("Total Load (kg)")
plt.title(f"Total Load on the Bridge on {selected_date}", fontsize=14, fontweight='bold', y=1.02)  
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# Save the figure
sum_plot_path = os.path.join(plot_directory, f"{selected_date}_Total_Load.png")
plt.savefig(os.path.join(plot_directory, f'{selected_date}_Summed_Load_Cell_Data.png'), format="png", dpi=300)
plt.show()

#%% Plot thermistor data and mean temp over defined windows
# Loop through each time range and calculate the mean temperature
mean_temperatures = []
time_labels = []

for i, (start, end) in enumerate(time_ranges):
    # Filter data within the time window
    mask = (temp_timestamps_datetime >= start) & (temp_timestamps_datetime < end)
    filtered_time = temp_timestamps_datetime[mask]
    filtered_data = temp_data_multi[mask]  # Use multi-sensor temp data

    # Compute mean temperature over selected thermistors
    if filtered_data.size > 0:
        mean_temp_per_window = np.mean(filtered_data, axis=1)  # Mean across thermistors at each timestamp
        overall_mean = np.mean(mean_temp_per_window)  # Single mean value for the time window

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_time, mean_temp_per_window, label=f"Mean Temperature", color='tab:blue')
        plt.axhline(overall_mean, color='red', linestyle='--', label=f"Overall Mean: {overall_mean:.2f} °C")

        plt.xlabel('Timestamp')
        plt.ylabel('Mean Temperature (°C)')
        plt.title(f"Mean Thermistor Temperature: {start.strftime('%H:%M')} - {end.strftime('%H:%M')} on {selected_date}", fontsize=14, fontweight='bold', y=1.02)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_filename = os.path.join(temp_plot_directory, f"{selected_date}_Mean_Temperature_{start.strftime('%H%M')}_{end.strftime('%H%M')}.png")
        plt.savefig(plot_filename, dpi=300)

        plt.show()

    else:
        print(f"No data available for time range: {start} - {end} on {selected_date}")


#%% Plot Dual Axis Normalized Accelerometer vs Mean Thermistor Data for Each Time Period
# Debugging alignment check
if (acc_timestamps_datetime[:5].equals(temp_timestamps_datetime[:5]) and 
    acc_timestamps_datetime[:5].equals(load_timestamps_datetime[:5])):
    print("Timestamps Align")
else:
    print("Timestamps Do Not Align")

# Loop through all 17 time windows
for start, end in time_ranges:
    acc_mask = (acc_timestamps_datetime >= start) & (acc_timestamps_datetime < end)
    temp_mask = (temp_timestamps_datetime >= start) & (temp_timestamps_datetime < end)

    # Filter accelerometer data
    acc_filtered_time = acc_timestamps_datetime[acc_mask]
    acc_filtered_data = acc_data_normalized[acc_mask]

    # Filter thermistor data (using all selected thermistors)
    temp_filtered_time = temp_timestamps_datetime[temp_mask]
    temp_filtered_data = temp_data_multi[temp_mask]  # Multi-sensor temp data

    # Compute mean temperature across selected thermistors
    if temp_filtered_data.size > 0:
        mean_temp_per_window = np.mean(temp_filtered_data, axis=1)  # Mean across thermistors at each timestamp
        overall_mean_temp = np.mean(mean_temp_per_window)  # Single mean value for the time window

        
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot normalised accelerometer data (y- axis)
    acc_line, = ax1.plot(acc_filtered_time, acc_filtered_data, label=f"Accelerometer: {selected_acc_channel_name}", color='b')
    ax1.set_xlabel("Timestamp", fontsize=12)
    ax1.set_ylabel("Acceleration (m/s²)", color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot thermsitor data (secondary y-axis)
    ax2 = ax1.twinx()
    temp_line, = ax2.plot(temp_filtered_time, mean_temp_per_window, label="Mean Thermistor Temp", color='r')
    mean_line = ax2.axhline(overall_mean_temp, color='red', linestyle='--', label=f"Mean Temp: {overall_mean_temp:.2f} °C")
    
    # Legend
    lines = [acc_line, temp_line, mean_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)
    
    plt.title(f"Normalised Accelerometer and Mean Thermistor Temperature\n{start.strftime('%H:%M')} - {end.strftime('%H:%M')} on {selected_date}\n{selected_acc_channel_name}",
              fontsize=14, fontweight='bold', y=1.02)
    
    ax1.grid(True)
    fig.tight_layout()

#%% Random Decrement Function
def calculate_random_decrement(acc_data_normalized, window_size=700, trigger_threshold=0.05):
    threshold = np.median(acc_data_normalized)  
    accumulated_sum = np.zeros(window_size, dtype=np.float64)  
    count = 0

    for i in range(len(acc_data_normalized) - window_size + 1):
        if acc_data_normalized[i] > threshold:  
            temp_window = acc_data_normalized[i:i + window_size].astype(np.float64)  
            accumulated_sum += temp_window
            count += 1

    if count > 0:
        rdsig = accumulated_sum / count  
    else:
        rdsig = np.zeros(window_size)  
    return rdsig

 
# Generate random decrement signature
fu = 200 # Frequency in Hz

# Compute RDS fucntion
rdsig = calculate_random_decrement(acc_filtered_data_normalized, window_size=700)
        

#%% Modal Parameter Estimation Function
def fmatpen_pandas(y,fs,K_min,K_max,n_excl=0,dataset = 0,combination = 0, acc=0):
    """
    Implementation of Matrix Pencil organized for use with Pandas timestep
    Calculates fitted signals for multiple model orders between K_min and K_max
    assuming RDS is comprised of free-decays of a linear system.
 
    Parameters
    ----------
    y : 1D Numpy array
        Random decrement signature to be analysed.
    fs : float
        Sample rate (Hz) of RDS.
    K_min : Int
        Minimum model order to use for fitting of RDS.
    K_max : Int
        Maximum model order to use for fitting of RDS.
    n_excl : Int, optional
        Number of samples WHICH HAVE ALREADY BEEN excluded from the start of the RDS.
        Used to account for excluded values in calculation of phase angle. The default is 0.
    dataset : Int, optional
        The reference number for the accelerometer . The default is 0.
    trigger_dataset : Int, optional
        The reference number for the accelerometer used as the trigger channel
        for generation of RDS. The default is 0.
    timestep : Int, optional
        Reference number for timestep relative to start of dataset. The default is 0.
 
    Returns
    -------
    modal_est : Numpy array
        Array containing modal estimates and key tracking data. From left to right the
        columns are: Timestep, trigger dataset, dataset, model orders, natural frequency (Hz),
        damping (% of critical), amplitude (relative), phase (radians), normalized amplitude
        (amplitude divided by maximum amplitude of RDS).
    y : the input signal
    y_f : the fitted signal
 
    """
    # Form hankel matrix
    N = len(y)
    ind = np.arange(N) + n_excl
    ls=int(np.floor(N/3))
    c = y[:N-ls]
    r = y[N-ls-1:N]
    X = linalg.hankel(c,r)
    # Application of Singular Value Decomposition:
    U, S, V = linalg.svd(X[:,1:ls+1], full_matrices=False)
 
    # Calculate number of modal estimates to be generated using triangular numbers
    n_terms = int(((K_max*(K_max+1))/2)-((K_min*(K_min-1))/2))
 
    # Initiate empty modal estimates array
    modal_est = np.zeros((n_terms,10))
    modal_est[:,0] = dataset # Column 0 = Timestep
    modal_est[:,1] = combination # Column 1 = Triggering dataset
    modal_est[:,2] = acc # Column 2 = Dataset
 
    n = 0 # n used as a counter
    # nk=0
    K = K_min
    r2 = 0
    while (K<=K_max) and (r2<0.99):
 
        modal_est[n:n+K,3] = K # Column 3 = Model order
        # Use try/except as the pseudo-inverse will occasionally fail
        try:
            # Use eigenvalues to calculate modal parameters of singular values
            p = np.log(linalg.eigvals(np.diag(1./S[:2*K]).dot((U[:,:2*K].T.dot(X[:,:ls])).dot(np.transpose(V)[:,:2*K]))))
            # Seperate eigenvalue into frequency and damping
            Om = np.imag(p)
            D = np.real(p)
            # Generate an array containing components of RDS
            Z=np.exp((-D.reshape(-1,1)*(ind))+(complex(0,1)*Om.reshape(-1,1)*(ind)))/2
            # Use pseudoinverse to calculate amplitude and phase of components of RDS
            R = np.array(np.linalg.pinv(Z.T)*np.matrix(y).H).squeeze()
            y_f =np.sum(np.real(Z*R.reshape(-1,1)),axis=0)
            # y_rd = y_f
            
            # Sort frequency according to frequency, keep only the first half of results
            # Note: This is similiar to keeping only positive results but with overfitting of signal
            # the number of positive and negative components may no longer be equal
            # Also ensures that the correct number of components are always taken
            indx = np.argsort(Om)[::-1][:K]
            modal_est[n:n+K,4] = Om[indx] # Column 4 = Natural frequency (radians)
            
            # Apply minimum threshold to avoid division by zero or near-zero frequencies
            modal_est[n:n+K,4] = np.where(modal_est[n:n+K,4] < 1e-6, 1e-6, modal_est[n:n+K,4])

            # Calculate damping ratio safely
            modal_est[n:n+K,5] = D[indx]/modal_est[n:n+K,4]  # Column 5 = Damping (% critical damping)

            
            modal_est[n:n+K,5] = D[indx]/modal_est[n:n+K,4] # Column 5 = Damping (% of critical damping) - We have to scale relative to the natural frequency
            modal_est[n:n+K,6] = np.abs(R)[indx] # Column 6 = Amplitude of component
            modal_est[n:n+K,7] = np.angle(R)[indx] # Column 7 = Phase of component
            modal_est[n:n+len(indx),9] = r2_score(y,y_f)
            r2 = modal_est[n,9]
        except:
            0
        n+=K
        K+=1
        # nk+=1
    modal_est[:,8] = modal_est[:,6]/np.max(abs(y)) # Column 8 = Normalized amplitude (divide by max absolute amplitude of RDS)
    modal_est[:,4]=modal_est[:,4]*fs/(2*np.pi) # Convert frequency from radians to Hz
    modal_est[:,5] = modal_est[:,5]*100 # Convert damping from fraction to percentage of critical damping
    return modal_est,y,y_f
 
sampling_frequency = 200  # Hz
max_model_order = 10 # It won't always go up to this order if the r2<0.99 condition is exceeded
mode_est,y,y_f=fmatpen_pandas(rdsig,sampling_frequency,0,max_model_order,n_excl=0,dataset = 0,combination = 0, acc=0)

#%% Process each 5-minute window
for start, end in time_ranges:
    acc_mask = (acc_timestamps_datetime >= start) & (acc_timestamps_datetime < end)
    acc_filtered_time = acc_timestamps_datetime[acc_mask]
    acc_filtered_data = acc_data[acc_mask]
    
    if acc_filtered_data.size > 0:
        # Normalize data for each 5-minute window
        acc_filtered_data_normalized = normalize_data(acc_filtered_data)

        # Compute RDS
        rdsig = calculate_random_decrement(acc_filtered_data_normalized, window_size=700)

        # Apply Matrix Pencil Method on rdsig[5:500]
        sampling_frequency = 200  
        max_model_order = 10 # It won't always go up to this order if the r2<0.99 condition is exceeded
        mode_est, _, best_fitted_curve = fmatpen_pandas(rdsig[5:700], sampling_frequency, K_min=1, K_max=10)

        # Create a single figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        # Set a global title for the figure
        fig.suptitle(f"Normalised Accelerometer Data and RDS for {start.strftime('%H:%M')} - {end.strftime('%H:%M')} on {selected_date}\n{selected_acc_channel_name}", 
                     fontsize=14, fontweight='bold', y=0.96)  # Adjust y to move title slightly
        # Plot Normalized Accelerometer Data
        axes[0].plot(acc_filtered_time, acc_filtered_data_normalized, label="Normalised Data")
        axes[0].set_xlabel("Timestamp")
        axes[0].set_ylabel("Acceleration (m/s²)")
        axes[0].set_title("Normalised Accelerometer Data")
        axes[0].legend()
        axes[0].grid(True)
        axes[0].tick_params(axis='x', rotation=45)

        # Plot RDS and Fitted Curve
        axes[1].plot(np.arange(5, 700), rdsig[5:700], label="RDS", color='r')
        axes[1].plot(np.arange(5, 700), best_fitted_curve[:695], linestyle="--", label="Fitted Curve (fmatpen)", color="b")
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Acceleration (m/s²)")
        axes[1].set_title("Random Decrement Signature & Fitted Curve")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(os.path.join(plot_directory, f"{selected_date}_{selected_acc_channel_name}_Normalised_Acc_&_RDS_{start.strftime('%H:%M')}_{end.strftime('%H:%M')}.png"), dpi=300)
        plt.show()

    else:
        print(f"No data available for time range: {start} - {end} on {selected_date}")

#%% Stabilisation Diagram for Modal Analysis with Centralized RDS
SD_results = []

# Loop through each 5-minute time range
for i, (start, end) in enumerate(time_ranges):
    # Filter accelerometer data for the current time range
    acc_mask = (acc_timestamps_datetime >= start) & (acc_timestamps_datetime < end)
    acc_filtered_data = acc_data_normalized[acc_mask]

    if acc_filtered_data.size == 0:
        print(f"No data available for time range: {start} - {end} on {selected_date}")
        continue

    # Compute the Random Decrement Signature (RDS)
    rdsig = calculate_random_decrement(acc_filtered_data, window_size=700)
    
    # Centralize RDS around zero
    rdsig_centralized = rdsig - np.mean(rdsig)

    # Apply Matrix Pencil Method (RDS[5:500])
    sampling_frequency = 200  
    mode_est, y, y_f = fmatpen_pandas(rdsig_centralized[5:700], sampling_frequency, K_min=1, K_max=10)

    # Extract modal data
    natural_frequencies = mode_est[:, 4]  # Column 4 = Natural Frequency (Hz)
    damping_ratios = mode_est[:, 5]       # Column 5 = Damping Ratio (%)
    model_orders = mode_est[:, 3]         # Column 3 = Model Order
    amplitudes = mode_est[:, 6]           # Column 6 = Amplitudes / Residual 

    # Filter out invalid frequencies (<1 Hz)
    valid_indices = natural_frequencies > 1
    natural_frequencies = natural_frequencies[valid_indices]
    damping_ratios = damping_ratios[valid_indices]
    model_orders = model_orders[valid_indices]
    amplitudes = amplitudes[valid_indices]

    if natural_frequencies.size == 0:
        print(f"No valid natural frequencies above 1Hz for time range: {start} - {end} on {selected_date}")
        continue

    # Identify the most prominent frequency (highest amplitude)
    max_amplitude_index = np.argmax(amplitudes)
    most_prominent_frequency = natural_frequencies[max_amplitude_index]
    corresponding_damping_ratio = damping_ratios[max_amplitude_index]
    max_amplitude = amplitudes[max_amplitude_index]

    # Get mean temperature in the time window
    temp_mask = (temp_timestamps_datetime >= start) & (temp_timestamps_datetime < end)
    filtered_temp_data = temp_data_multi[temp_mask]
    
    if filtered_temp_data.size == 0:
        print(f"No temperature data available for {start} - {end} on {selected_date}")
        continue
    
    mean_temp = np.mean(filtered_temp_data)

    # Store results
    SD_results.append({
        "Time Range": f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}",
        "Most Prominent Frequency (Hz)": most_prominent_frequency,
        "Damping Ratio (%)": corresponding_damping_ratio,
        "Amplitude": max_amplitude,
        "Mean Temperature (°C)": mean_temp
    })

    # Create a single figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Set a global title for the entire figure
    fig.suptitle(f"Modal Analysis with Centralised RDS - {selected_acc_channel_name}\n{start.strftime('%H:%M')} - {end.strftime('%H:%M')} on {selected_date}",
                 fontsize=14, fontweight='bold', y=0.92)

    # Plot Centralized RDS
    axes[0].plot(np.arange(5, 700), rdsig_centralized[5:700], label="Centralised RDS", color='r')
    axes[0].plot(np.arange(5, 700), y_f[:695], linestyle="--", label="Fitted RDS (fmatpen)", color="b")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Random Decrement Signature")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Stabilization Diagram
    sc = axes[1].scatter(natural_frequencies, model_orders, c=amplitudes, cmap="viridis", s=50)
    fig.colorbar(sc, ax=axes[1], label="Amplitude")  # Add colorbar
    axes[1].set_xlabel("Natural Frequency (Hz)")
    axes[1].set_ylabel("Model Order")
    axes[1].set_title("Stabilisation Diagram")
    axes[1].grid(True)

    # Adjust layout to leave space for the title
    plt.tight_layout(rect=[0, 0, 1, 0.93])  

    # Save both plots in **one image**
    save_path = os.path.join(plot_directory, f"{selected_date}_{selected_acc_channel_name}_Modal_Analysis_{start.strftime('%H%M')}_{end.strftime('%H%M')}.png")
    plt.savefig(save_path, dpi=300)

    # Show the figure with both plots
    plt.show()

# Convert results to a DataFrame and print
SD_results_df = pd.DataFrame(SD_results)
print(SD_results_df)

#%% Plot Highest Amplitude Frequency vs. Mean Temperature (With Timestamp Labels) with REGRESSION LINES

# Filter to keep only positive damping ratio values and frequencies between 5 Hz and 7 Hz
valid_indices = (SD_results_df["Most Prominent Frequency (Hz)"] >= 5) & \
                (SD_results_df["Most Prominent Frequency (Hz)"] <= 7) & \
                (SD_results_df["Damping Ratio (%)"] > 0)

# Apply the mask to filter the data
x_temp_filtered = SD_results_df["Mean Temperature (°C)"][valid_indices]
y_freq_filtered = SD_results_df["Most Prominent Frequency (Hz)"][valid_indices]
y_damping_filtered = SD_results_df["Damping Ratio (%)"][valid_indices]
amplitudes_filtered = SD_results_df["Amplitude"][valid_indices]

# Perform linear regression only on filtered data
freq_slope, freq_intercept = np.polyfit(x_temp_filtered, y_freq_filtered, 1)  
damping_slope, damping_intercept = np.polyfit(x_temp_filtered, y_damping_filtered, 1)

# Generate regression lines
x_fit = np.linspace(min(x_temp_filtered), max(x_temp_filtered), 100)
y_freq_fit = freq_slope * x_fit + freq_intercept
y_damping_fit = damping_slope * x_fit + damping_intercept

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Set a global title for the figure
fig.suptitle(
    f"Regression Analysis of Temperature Effects on Frequency and Damping Ratio\n{selected_date}\n{selected_acc_channel_name}",
    fontsize=18, fontweight='bold', y=1
)

# Plot Most Prominent Frequency vs. Mean Temperature 
sc1 = axes[0].scatter(
    x_temp_filtered, y_freq_filtered, c=amplitudes_filtered, cmap="viridis", 
    s=80, edgecolors="black", label="Data"
)
axes[0].plot(x_fit, y_freq_fit, color="red", linestyle="--", linewidth=2, 
             label=f"y = {freq_slope:.3f}x + {freq_intercept:.3f}\n(Gradient: {freq_slope:.3f})")
fig.colorbar(sc1, ax=axes[0], label="Amplitude")

# Annotate points
for idx, row in SD_results_df[valid_indices].iterrows():
    axes[0].text(row["Mean Temperature (°C)"], row["Most Prominent Frequency (Hz)"], 
                 row["Time Range"], fontsize=10, ha="center", va="bottom")

axes[0].set_xlabel("Mean Temperature (°C)")
axes[0].set_ylabel("Most Prominent Frequency (Hz)")
axes[0].set_title(
    f"Most Prominent Frequency (5 - 7 Hz) vs. Mean Temperature",
    fontsize=14, y = 1.02)
axes[0].legend()
axes[0].grid(True)

# Plot Corresponding Damping Ratio vs. Mean Temperature
sc2 = axes[1].scatter(
    x_temp_filtered, y_damping_filtered, c=y_freq_filtered, cmap="plasma", 
    s=80, edgecolors="black", label="Data"
)
axes[1].plot(x_fit, y_damping_fit, color="red", linestyle="--", linewidth=2, 
             label=f"y = {damping_slope:.3f}x + {damping_intercept:.3f}\n(Gradient: {damping_slope:.3f})")
fig.colorbar(sc2, ax=axes[1], label="Most Prominent Frequency (Hz)")

# Annotate points
for idx, row in SD_results_df[valid_indices].iterrows():
    axes[1].text(row["Mean Temperature (°C)"], row["Damping Ratio (%)"], 
                 row["Time Range"], fontsize=10, ha="center", va="bottom")

axes[1].set_xlabel("Mean Temperature (°C)")
axes[1].set_ylabel("Damping Ratio (%)")
axes[1].set_title(
    f"Corresponding Positive Damping Ratio vs. Mean Temperature",
    fontsize=14, y = 1.02)
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(plot_directory, f'{selected_date}_{selected_acc_channel_name}_Regression_Analysis_of_Temperature_Effects_on_Frequency_and_Damping_Ratio_with_Colour_Grading.png'), format="png", dpi=300)
plt.show()

#%% Minimum Load vs. Mean Temperature with Natural Frequency Color Coding
# (at minumum load we can assume no people on the bridge)

mean_temps = []
min_loads = []
nat_freqs = []
time_labels = []

for start, end in time_ranges:
    # Temp Data
    temp_mask = (temp_timestamps_datetime >= start) & (temp_timestamps_datetime < end)
    temp_filtered = temp_data_multi[temp_mask]
    if temp_filtered.size == 0:
        continue
    mean_temp = np.mean(temp_filtered)
    
    # Laod Data
    load_mask = (load_timestamps_datetime >= start) & (load_timestamps_datetime < end)
    load_filtered = load_data[load_mask]
    if load_filtered.size == 0:
        continue
    min_load = np.min(load_filtered)
    
    # Store the values for plotting
    mean_temps.append(mean_temp)
    min_loads.append(min_load)
    # nat_freqs.append(most_prominent_freq)
    time_labels.append(f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}")

# Create the scatter plot with color-coded natural frequency
# Scatter Plot with Regression
plt.figure(figsize=(10, 6))

sc = plt.scatter(mean_temps, min_loads, s=100, edgecolors='k', color='skyblue', label='Data Points')
# Fit Line of Best Fit (Linear Regression)
X = np.array(mean_temps).reshape(-1, 1)
y = np.array(min_loads)
model = LinearRegression()
model.fit(X, y)
x_fit = np.linspace(min(mean_temps), max(mean_temps), 100).reshape(-1, 1)
y_fit = model.predict(x_fit)
# Plot best fit line
plt.plot(x_fit, y_fit, linestyle='--', color='red', linewidth=2,
         label=f"Best Fit: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

for i, label in enumerate(time_labels):
    plt.annotate(label, (mean_temps[i], min_loads[i]), textcoords="offset points", xytext=(5, 5), fontsize=10)


plt.xlabel("Mean Temperature (°C)", fontsize=12)
plt.ylabel("Minimum Load (kg)", fontsize=12)
plt.suptitle(f"Minimum Load vs. Mean Temperature\n{selected_date}\n{selected_load_channel_name}", 
             fontsize=16, fontweight="bold", y=0.92)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig(os.path.join(plot_directory, f'{selected_date}_{selected_load_channel_name}_Min_Load_vs_Mean_Temperature_with_BestFit.png'), format="png", dpi=300)
plt.show()

#%%  Save freq & damping csvs
save_dir = "/Users/lucymccoy/Desktop/thesis/coding/saved_dataframes"
os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

# Define save file path
save_file = os.path.join(save_dir, f"{selected_acc_channel_name}_{selected_date}_natural_freq_damping_vs_temp.csv")

# Prepare data for saving with Timestamp and Accelerometer Identification
df_to_save = SD_results_df[["Mean Temperature (°C)", "Most Prominent Frequency (Hz)", "Damping Ratio (%)", "Time Range"]].copy()
df_to_save.rename(columns={"Time Range": "Timestamp"}, inplace=True)  # Rename column for clarity

# Add a column to identify the accelerometer used
df_to_save["Accelerometer"] = selected_acc_channel_name

# Check if file exists to append or create a new file
if os.path.exists(save_file):
    existing_df = pd.read_csv(save_file)
    combined_df = pd.concat([existing_df, df_to_save], ignore_index=True)
else:
    combined_df = df_to_save

# Save updated DataFrame
combined_df.to_csv(save_file, index=False)

print(f"Data successfully saved to {save_file}")

