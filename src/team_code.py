#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from scipy.signal import resample

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.

def preprocess_data(data, sampling_frequency, utility_frequency, channels):
    # If data has less than 5 min return None
    if data.shape[1] < 5 * 60 * sampling_frequency:
        return None, sampling_frequency

    # If data is full of zeros or Nan return None
    if np.all(data == 0) or np.all(np.isnan(data)):
        return None, sampling_frequency

    # Apply filters
    passband = [0.1, 45.0]
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, verbose='error')

    
    data = data.astype(np.float64)

    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], verbose='error')

    # Resample the data
    resampling_frequency = 128
    num_samples = int(data.shape[1] * resampling_frequency / sampling_frequency)
    data = resample(data, num_samples, axis=1)

    # Scale the data
    if np.nanmin(data) != np.nanmax(data):
        data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data)
    else:
        data = 0 * data

    return pd.DataFrame(data.T, columns=channels, dtype="float32"), resampling_frequency


# Get bipolar data and sampling frecuency of a certain recording of a patient.
def get_bipolar_data(data):

    BIPOLAR_PAIRS = [
        ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),  # Left temporal chain
        ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),  # Right temporal chain
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),  # Left parasagittal chain
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),  # Right parasagittal chain
        ("Fz", "Cz"), ("Cz", "Pz")                                # Central chain
    ]
    
    bipolar_data = pd.DataFrame(np.full((data.shape[0], len(BIPOLAR_PAIRS)), np.nan))
    bipolar_names = []

    for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
        # Missing channel -> just NaN
        if ch1 not in data.columns or ch2 not in data.columns:
            continue
        bipolar_data.iloc[:, i] = data[ch1] - data[ch2]
        bipolar_names.append(f"{ch1}-{ch2}")

    bipolar_data.columns = bipolar_names
    return bipolar_data


# Extract features.
def get_features(data_folder, patient_id):
    EEG_CHANNELS = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4",
                    "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2",
                    "Fz", "Cz", "Pz", "Fpz", "Oz", "F9"]
    GROUP = 'EEG'

    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.

    eeg_features = float('nan') * np.ones(8)
    if num_recordings <= 0:
        return (patient_features, eeg_features)
    
    recording_id = recording_ids[-1]
    recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, GROUP))
    if not os.path.exists(recording_location + '.hea'):
        return (patient_features, eeg_features)
    
    #data, channels, sampling_frequency = load_recording_data(recording_location)
    utility_frequency = get_utility_frequency(recording_location + '.hea')
    
    # if not all(channel in channels for channel in EEG_CHANNELS):
    #    return (patient_features, eeg_features)
     
    #data = expand_channels(data, channels, EEG_CHANNELS)
    #data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
    #data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]]) # Convert to bipolar montage: F3-P3 and F4-P4
    #data = get_bipolar_data(data, channels, sampling_frequency)
    #print(data.shape)
    #eeg_features = get_eeg_features(data, sampling_frequency).flatten()

    
    return (patient_features, eeg_features)

# Extract patient features from the data.
def get_patient_features(data):
    hospital = get_hospital(data)
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    features = {
        'hospital': hospital,
        'age': age,
        'sex': sex,
        'rosc': rosc,
        'ohca': ohca,
        'sr': shockable_rhythm,
        'ttm': ttm
    }

    df = pd.DataFrame([features])

    return df

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features