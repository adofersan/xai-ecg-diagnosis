import numpy as np
import pandas as pd
from scipy.signal import hilbert
from line_profiler import profile
from datetime import time

def calc_nan_quality(EEG_df, seg_masks_df):
    total_values = EEG_df.size
    missing_values = EEG_df.isna().sum().sum()
    nan_quality = (1 - missing_values / total_values) * 100
    seg_masks_df["nan_quality"] = nan_quality

def calc_gap_quality(EEG_df, seg_masks_df):
    missing_percentages = EEG_df.isna().mean()
    best_channel = missing_percentages.idxmin()
    best_channel_data = EEG_df[best_channel].values
    
    # Compute the longest valid subsequence for the selected channel
    mask = ~np.isnan(best_channel_data)  # True for non-NaN, False for NaN
    diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]  # Start indices of valid sequences
    ends = np.where(diff == -1)[0]   # End indices of valid sequences

    gap_quality = np.max(ends - starts) if starts.size > 0 else 0  # Find the longest sequence length
    gap_quality = gap_quality / EEG_df.shape[0] * 100  # Normalize by the total number of samples
    seg_masks_df["gap_quality"] = gap_quality


def calc_outlier_quality(EEG_df, seg_masks_df, threshold=1.64):
    std_all = EEG_df.values.std()  # Desviación estándar global
    anomalies = np.abs(EEG_df) > (threshold * std_all)  # Matriz booleana de valores anómalos
    outlier_quality = 100 * (1 - anomalies.sum().sum() / EEG_df.size)  # Norm value prop

    # SHIFT = 80
    # outlier_quality = (100 / (100 - SHIFT)) * (outlier_quality - SHIFT)
    # outlier_quality = np.clip(outlier_quality, 0, 100)

    seg_masks_df["outlier_quality"] = outlier_quality


def calc_cohesion_quality(EEG_df, seg_masks_df):
    EEG_data = EEG_df.fillna(0).values.T
    phases = np.angle(hilbert(EEG_data, axis=1))
    plv_matrix = np.abs(np.exp(1j * (phases[:, np.newaxis, :] - phases[np.newaxis, :, :])).mean(axis=2))
    upper_tri = np.triu_indices_from(plv_matrix, k=1)
    cohesion_score = 100 * np.nanmean(plv_matrix[upper_tri])
    seg_masks_df["cohesion_quality"] = cohesion_score

def calc_flat_quality(EEG_df, seg_masks_df, Fs, flat_sec=5, cv_thres=0.01):
    # Compute rolling window size in samples
    window_size = int(round(flat_sec * Fs))
    
    # Compute rolling mean and std
    rolling = EEG_df.rolling(window=window_size, min_periods=1)  # Ensure it computes for small windows
    rolling_mean = rolling.mean()
    rolling_std = rolling.std()

    # Handle zero mean values properly
    rolling_mean[rolling_mean == 0] = np.nan

    # Compute coefficient of variation (CV)
    rolling_cv = rolling_std / rolling_mean

    # Define flat regions based on CV threshold
    flat_mask = rolling_cv <= cv_thres
    flat_percentage_per_channel = flat_mask.mean(axis=0, skipna=True) * 100
    flat_quality = 100 - flat_percentage_per_channel.mean()
    
    seg_masks_df["flat_quality"] = flat_quality


def calc_sharpness_quality(EEG_df, seg_masks_df, amplitude_thres):
    # Compute absolute differences (first row will be NaN)
    diff_signal = EEG_df.diff().abs()
    # Compute 95th percentile of absolute EEG values for each channel
    signal_percentile = EEG_df.abs().quantile(0.95, axis=0)
    # Detect large changes (preserve NaNs)
    large_changes = diff_signal >= amplitude_thres * signal_percentile
    large_changes = large_changes.where(~diff_signal.isna(), np.nan)  # Keep NaNs
    # Count large changes per channel, ignoring NaNs
    num_large_changes = large_changes.sum(axis=0, skipna=True)

    # Compute valid sample count per channel (excluding NaNs)
    valid_samples_per_channel = large_changes.notna().sum(axis=0)

    # Compute average large changes per valid sample across all channels
    prop_large_changes = (num_large_changes / valid_samples_per_channel)
    avg_large_changes = prop_large_changes.mean()
    sharpness_quality = 100 * (1 - avg_large_changes)
    seg_masks_df["sharpness_quality"] = sharpness_quality


# def find_muscle_artifact(EEG_df, Fs, seg_masks_df, threshold=3.0):
#     psd, _ = mne.time_frequency.psd_array_welch(EEG_df.T, sfreq=Fs, fmin=30, fmax=60)
#     high_freq_power = np.mean(psd, axis=1)
#     z_scores = (high_freq_power - np.mean(high_freq_power)) / np.std(high_freq_power)
#     if np.any(z_scores > threshold):
#         seg_masks_df["emg_contamination"] = 1

@profile
def process_EEG(EEG_df, Fs, patient_id, start_time, end_time):
    seg_masks_df = pd.DataFrame({
        "patient_id": [patient_id],
        "start_time": [start_time],
        "end_time": [end_time],

        "nan_quality": [0],
        "gap_quality": [0],
        "outlier_quality": [0],
        "flat_quality": [0],
        "sharpness_quality": [0],
        "cohesion_quality": [0]

        #"emg_contamination": [0]
    })
    
    calc_nan_quality(EEG_df, seg_masks_df)
    calc_gap_quality(EEG_df, seg_masks_df)
    calc_outlier_quality(EEG_df, seg_masks_df, threshold=2)
    calc_flat_quality(EEG_df, seg_masks_df, Fs, flat_sec=5, cv_thres=0.01)
    calc_sharpness_quality(EEG_df, seg_masks_df, amplitude_thres=0.05)
    calc_cohesion_quality(EEG_df, seg_masks_df)
    #find_muscle_artifact(EEG_df, Fs, seg_masks_df)
    
    return seg_masks_df


def main():
    channel_num, sample_num, Fs = 18, 38400, 128
    EEG_df = pd.DataFrame(np.random.randn(sample_num, channel_num))
    patient_id = "P12345"
    start_time = time(23, 0, 0)
    end_time = time(23, 59, 59)
    
    seg_masks_df = process_EEG(EEG_df, Fs, patient_id, start_time, end_time)
    
    print("Segment Masks DataFrame:")
    print(seg_masks_df)

if __name__ == "__main__":
    main()
