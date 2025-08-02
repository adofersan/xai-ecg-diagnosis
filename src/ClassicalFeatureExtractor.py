import numpy as np
import pandas as pd
import antropy as ant
from scipy.signal import welch
from scipy.integrate import simpson
from neurokit2 import fractal_dfa
from line_profiler import profile

from scipy.stats import entropy


class ClassicalFeatureExtractor:
    def __init__(self, df, sf=128):
        """
        Initialize the ClassicalFeatureExtractor class.
        :param df: pandas DataFrame (columns as EEG channels)
        :param sf: Sampling frequency of EEG data (default: 128 Hz)
        """
        self.df = df
        self.sf = sf

    def compute_entropy_features(self, data):
        try:
            entropy_values = {
                # "ENT_ApEn": ant.app_entropy(data, order=2),
                # "ENT_SampEn": ant.sample_entropy(data, order=2), deleted beacuse of high computation time
                # "ENT_shannon": entropy(data),
                "ENT_perm": ant.perm_entropy(data, normalize=True),
                "ENT_spectral": ant.spectral_entropy(
                    data, sf=self.sf, method="welch", normalize=True
                ),
                "ENT_svd": ant.svd_entropy(data, normalize=True),
            }
        except Exception as e:
            # Log the exception if needed, e.g., print(f"Error computing entropy: {e}")
            entropy_values = {
                "ENT_perm": -1,
                "ENT_spectral": -1,
                "ENT_svd": -1,
            }
        return entropy_values

    def compute_fractal_features(self, data):
        try:
            fractal_values = {
                # "FRC_DFA": ant.detrended_fluctuation(data),
                "FRC_higuchi": ant.higuchi_fd(data),
                "FRC_petrosian": ant.petrosian_fd(data),
                "FRC_katz": ant.katz_fd(data),
            }

        except Exception as e:
            fractal_values = {
                "FRC_higuchi": -1,
                "FRC_petrosian": -1,
                "FRC_katz": -1,
            }
        return fractal_values

    def compute_spectral_features(self, data):

        FREQ_BANDS = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

        f, Pxx = welch(data, fs=self.sf, nperseg=4 * self.sf)
        power = {
            band: simpson(Pxx[(f >= low) & (f < high)], f[(f >= low) & (f < high)])
            for band, (low, high) in FREQ_BANDS.items()
        }
        total_power = sum(power.values())
        total_power = max(total_power, 1e-10)  # Avoid division by zero

        rel_power = {"SPC_" + band: pwr / total_power for band, pwr in power.items()}

        # Spectral Edge Frequency 95%
        # sef95 = f[np.cumsum(Pxx) / np.sum(Pxx) >= 0.95][0]
        # fsi = (power["gamma"] + power["beta"]) / (power["delta"] + power["theta"])

        return {**rel_power}

    # def compute_background_indices(self, data):
    #     psds, freqs = welch(data, fs=self.sf, nperseg=4 * self.sf)

    #     bci_values = []
    #     for i in range(18):
    #         psd = psds[i, :]
    #         bci = np.var(psd)
    #         bci_values.append(bci)

    #     threshold = 2.0
    #     burst_amplitude = []
    #     suppressed_amplitude = []
    #     for i in range(18):
    #         eeg_channel = raw.get_data()[i, :]

    #         # Burst detection: find where the signal exceeds the threshold
    #         bursts = eeg_channel > threshold
    #         suppressed = eeg_channel < -threshold

    #         # Calculate burst and suppressed amplitudes
    #         burst_amplitude.append(np.mean(eeg_channel[bursts]))
    #         suppressed_amplitude.append(np.mean(eeg_channel[suppressed]))

    #     # Compute BSAR (burst amplitude / suppressed amplitude)
    #     bsar_values = np.array(burst_amplitude) / np.array(suppressed_amplitude)

    #     return {"BGI_bci": bci, "BGI_bsar": bsar_values}

    def extract_all_features(self):
        feature_list = self.df.apply(
            lambda col: {
                **self.compute_entropy_features(col.values),
                **self.compute_fractal_features(col.values),
                **self.compute_spectral_features(col.values),
                # **self.compute_background_indices(col.values),
            },
            axis=0,
        )

        # Convert extracted features into a DataFrame
        feature_df = pd.DataFrame(feature_list.tolist(), index=self.df.columns)

        # Flatten the DataFrame so that each feature has a unique column
        feature_flat = feature_df.stack().to_frame().T
        feature_flat.columns = [
            f"{feature}_{channel}" for channel, feature in feature_flat.columns
        ]

        # Compute summary statistics
        summary_stats = feature_df.aggregate(["mean", "std"])

        # Rename summary statistics columns with the channel name at the end
        summary_stats_flat = summary_stats.unstack().to_frame().T
        summary_stats_flat.columns = [
            f"{feature}_{stat}" for feature, stat in summary_stats_flat.columns
        ]

        # Combine individual feature values and summary statistics
        final_df = pd.concat([feature_flat, summary_stats_flat], axis=1)

        return final_df

    # @profile  # Profiling on the main function
    # def extract_all_features2(self, patient_id):
    #     feature_list = []

    #     for col in self.df.columns:
    #         # Start profiling each function separately

    #         # Start timer for entropy features
    #         entropy_features = self.compute_entropy_features(self.df[col].values)

    #         # Start timer for spectral features
    #         spectral_features = self.compute_spectral_features(self.df[col].values)

    #         # Start timer for background indices
    #         background_features = self.compute_background_indices(self.df[col].values)

    #         # Start timer for nonlinear features
    #         nonlinear_features = self.compute_fractal_features(self.df[col].values)
    #     return None


if __name__ == "__main__":
    np.random.seed(42)
    # 5 min has 30 epochs
    for i in range(6 * 5):
        num_channels = 18
        num_samples = 1280
        random_data = pd.DataFrame(
            np.random.randn(num_samples, num_channels),
            columns=[f"Ch{i+1}" for i in range(num_channels)],
            dtype=np.float32,
        )

        eeg_extractor = ClassicalFeatureExtractor(random_data)
        features_df = eeg_extractor.extract_all_features()
        # print(features_df)
        features_df.to_csv("CLASSICAL_features.csv", index=False)
        break
