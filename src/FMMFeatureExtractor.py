
from math import pi
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft
from sklearn.linear_model import LinearRegression
from line_profiler import profile

class FMMFeatureExtractor:
    def __init__(self, data) -> None:
        self.data = data.T.to_numpy()
        self.columns = data.columns
        self.counter = None

    @staticmethod
    def complex_transform(x, t):
        return ((1 - np.abs(x) ** 2) ** 0.5) / (1 - np.conj(x) * np.exp(1j * t))

    @staticmethod
    def calculate_coefficient(a: complex, t: Iterable[Iterable[float]], G: Iterable[complex], n_obs: int):
        denominator = (1 - np.conj(a) * np.exp(1j * t)).dot(G.conj().T)
        if denominator == 0:
            denominator = 0.000001 + 0.000001j
        return np.conj(((1 - np.abs(a) ** 2) ** 0.5) / denominator) / n_obs

    def generate_circle_disk(self, max_magnitude: float, n_obs: int, max_phase: float) -> np.ndarray:
        phase = np.arange(0.0, max_phase + 2 * np.pi / n_obs, 2 * np.pi / n_obs)[np.newaxis, :]
        magnitude = np.sort(-np.arange(np.sqrt(0.01), np.sqrt(0.6), ((np.sqrt(0.6) - np.sqrt(0.01)) / 24)) ** 2 + 1)[np.newaxis, :]
        n_phase = phase.shape[1]
        n_magnitude = magnitude.shape[1]
        magnitude = np.repeat(magnitude, n_phase, axis=0)
        phase = np.repeat(phase.T, n_magnitude, axis=1)

        disk = magnitude * np.exp(1j * phase)
        disk[np.abs(disk) - max_magnitude >= -1e-15] = np.nan

        disk = disk[~np.isnan(disk).all(axis=1)]
        disk = disk[:, ~np.isnan(disk).all(axis=0)]

        return disk

    def update_channel(self, n_channels, n_obs, coefficients, residuals, an, t, component):
        for ch in range(n_channels):
            coefficients[ch, component] = FMMFeatureExtractor.calculate_coefficient(an[component], t, residuals[ch, :], n_obs)[0]
            residuals[ch, :] = (
                (residuals[ch, :] - coefficients[ch, component] * FMMFeatureExtractor.complex_transform(an[component], t))
                * (1 - np.conj(an[component]) * np.exp(1j * t))
                / (np.exp(1j * t) - an[component])
            )

    def get_fft_base(self, dic_an, t, n_obs):
        self.counter += 1
        return fft(FMMFeatureExtractor.complex_transform(dic_an[0, self.counter], t), n_obs)

    def calculate_amplitudes_betas(self, n_channels, n_components, t, alphas2, omegas):
        amplitudes = np.zeros((n_channels, n_components))
        betas = np.zeros((n_channels, n_components))
        mm = np.zeros((len(t[0]), 2 * n_components + 1))
        mm[:, 0] = np.ones(len(t[0]))
        for i in range(n_components):
            t_star = 2 * np.arctan(omegas[i] * np.tan((t - alphas2[i]) / 2))
            mm[:, 2 * (i + 1) - 1] = np.cos(t_star)
            mm[:, 2 * (i + 1)] = np.sin(t_star)

        mm = np.linalg.pinv(mm.T @ mm) @ mm.T
        for ch in range(n_channels):
            coefs = mm @ self.data[ch, :]
            for i in range(n_components):
                amplitudes[ch, i] = np.sqrt(coefs[2 * i + 1] ** 2 + coefs[2 * i + 2] ** 2)
                betas[ch, i] = np.arctan2(-coefs[2 * i + 2], coefs[2 * i + 1])

        betas = np.mod(betas, 2 * np.pi)
        return amplitudes, betas

    def calculate_r2_index(self, n_channels, n_components, t, alphas2, omegas, betas, residuals):
        R2comp = np.zeros((n_channels, n_components + 1))
        for component in range(n_components):
            for ch in range(n_channels):
                cos_phi = np.cos(betas[ch, component] + 2 * np.arctan(omegas[component] * np.tan((t - alphas2[component]) / 2)))
                
                model = LinearRegression(n_jobs=1)
                model.fit(cos_phi.reshape(-1, 1), residuals[ch, :].reshape(-1, 1))
                
                intercept = model.intercept_[0]
                coef = model.coef_[0, 0]
                
                # Fixing the broadcasting error
                residuals[ch, :] -= (intercept + coef * cos_phi).flatten()  # Flatten to ensure shape (1280,)
                
                variance = np.var(self.data[ch, :])
                if variance == 0:
                    variance = 0.00001
                R2comp[ch, component + 1] = 1 - np.var(residuals[ch, :]) / variance

        R2comp = np.mean(np.diff(R2comp, axis=1), axis=0)
        R2comp = R2comp / np.sum(R2comp)
        return R2comp

    def estimate_parameters(self, n_components: int = 10) -> pd.DataFrame:
        analytic_signal = signal.hilbert(self.data)
        n_channels, n_obs = analytic_signal.shape
        t = np.expand_dims(np.arange(0, n_obs) / n_obs * 2 * pi, axis=0)

        dic_an = self.generate_circle_disk(1, n_obs, 0)
        dic_an_search = self.generate_circle_disk(1, n_obs, 2 * pi - 2 * pi / n_obs)
        _, an_search_len = dic_an_search.shape
        base = np.zeros((an_search_len, n_obs))
        self.counter = -1
        base = np.apply_along_axis(lambda row: self.get_fft_base(dic_an, t, n_obs), 1, base)
        base = base.reshape((an_search_len, n_obs))

        an = np.zeros((n_components + 1), "complex")
        coefficients = np.zeros((n_channels, n_components + 1), "complex")

        residuals = analytic_signal
        self.update_channel(n_channels, n_obs, coefficients, residuals, an, t, 0)

        for component in range(1, n_components + 1):
            S1_tmp = np.zeros((an_search_len, n_obs))
            for ch in range(n_channels):
                S1_tmp += np.abs(ifft(np.repeat(fft(residuals[ch, :], n_obs)[np.newaxis, :], an_search_len, axis=0) * base, n_obs, 1))

            S1_tmp = S1_tmp.T
            max_loc_tmp = np.argwhere(S1_tmp == np.amax(S1_tmp))
            an[component] = dic_an_search[max_loc_tmp[0, 0], max_loc_tmp[0, 1]]
            self.update_channel(n_channels, n_obs, coefficients, residuals, an, t, component)

        alphas = np.angle(an)
        alphas = np.unwrap(alphas)[1:]
        omegas = (1 - np.abs(an)) / (1 + np.abs(an))
        omegas = omegas[1:]

        alphas2 = np.mod(alphas + np.pi, 2 * np.pi)
        amplitudes, betas = self.calculate_amplitudes_betas(n_channels, n_components, t, alphas2, omegas)

        residuals = self.data.copy()
        R2comp = self.calculate_r2_index(n_channels, n_components, t, alphas2, omegas, betas, residuals)

        time_order = np.argsort(alphas)
        alphas = alphas[time_order]
        alphas2 = alphas2[time_order]
        omegas = omegas[time_order]
        amplitudes = amplitudes[:, time_order].T
        betas = betas[:, time_order]
        R2comp = R2comp[time_order]

        column_names = ["R2", "α", "ω"] + ["A_" + c for c in self.columns]
        column_names = ["FMM_" + x for x in column_names]
        fmm_params = pd.DataFrame(
            np.column_stack((R2comp, alphas2, omegas, amplitudes)),
            columns=column_names,
        )
        fmm_params.insert(0, "Wave", np.arange(1, n_components + 1))
        # compute median across rows, but as a DataFrame
        #fmm_params = pd.DataFrame(fmm_params.median(axis=0)).transpose()

        # fmm_params = fmm_params.assign(median_a=np.median(amplitudes, axis=1))
        # a_std = np.std(amplitudes, axis=1)
        # a_std[a_std == 0] = np.nan
        # fmm_params = fmm_params.assign(cv_a=np.nan_to_num(np.mean(amplitudes, axis=1) / a_std))


        return fmm_params


    
@profile
def main():

    # 5 min has 30 epochs
    for i in range(6*5):
        n_channels = 18
        n_obs = 1280

        np.random.seed(42)
        data = np.random.rand(n_channels, n_obs)

        fmm_extractor = FMMFeatureExtractor(data)
        fmm_params = fmm_extractor.estimate_parameters(n_components=10)
        fmm_params.to_csv("FMM_features.csv", index=False)
        #print(fmm_params)
        break
        

if __name__ == "__main__":
    main()