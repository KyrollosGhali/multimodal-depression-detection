import numpy as np
from scipy.signal import butter, filtfilt, resample
import pyedflib
import pickle
import joblib
import cv2
from skimage.feature import hog
from scipy.signal import stft
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
geneva_order = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
        'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    ]
def compute_stft(signal, fs=128, nperseg=128):
    """
    Compute STFT of a single EEG channel.
    :param signal: 1D array of EEG samples
    :param fs: Sampling frequency (e.g., 128 Hz for DEAP)
    :param nperseg: Window size for STFT (default 1 second)
    :return: Time-frequency magnitude spectrogram
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    return np.abs(Zxx)
def create_efdm(eeg_trial):
    """
    Generate EFDM for a single EEG trial.
    :param eeg_trial: EEG data for one trial (32 channels, samples)
    :return: EFDM image
    """
    freq_bins, time_bins = 65, 63  # Assume STFT shape
    efdm = np.zeros((5, 8, freq_bins))  # 5x5 spatial map with frequency info
    electrode_positions =  {f'ch{i}': (x, y) for i, (x, y) in enumerate((x, y) for x in range(4) for y in range(8))}
    for i, (electrode, pos) in enumerate(electrode_positions.items()):
        stft_result = compute_stft(eeg_trial[i])  # STFT of channel
        avg_power = np.mean(stft_result, axis=1)  # Average over time bins
        efdm[pos[0], pos[1], :] = avg_power  # Assign frequency distribution
    
    return efdm
def Extract_EEG_Features3(eeg_signals):
    """
    Extract HOG features from EEG signals for classification.
    :param eeg_signals: 2D array of EEG signals (channels x samples)
    :return: List of HOG features for each segment of EEG signals
    """
    features = []
    n_channels = eeg_signals.shape[0]
    if n_channels < 32:
        raise ValueError(f"Expected 32 channels, got {n_channels}. Check your EDF file and preprocessing.")
    if eeg_signals.shape[1]/128 > 63:
        period = int(eeg_signals.shape[1]/128)
        reminder = (eeg_signals.shape[1] % 128)%63
        for i in range(reminder, period+1, 63):
            x=create_efdm(eeg_signals[:,i*128:((i+63)*128)+1])
            x=cv2.resize(np.mean(x, axis=-1), (64, 64))
            hog_features = hog(
                x,
                orientations=9,  # Number of gradient orientations
                pixels_per_cell=(8, 8),  # Size of each cell
                cells_per_block=(2, 2),  # Blocks of cells for normalization
                block_norm='L2-Hys',  # Normalization method
                # visualize=visualize,  # Return HOG image if True
                transform_sqrt=True  # Improve contrast
            )
            features.append(hog_features)
    else:
        x=create_efdm(eeg_signals[:,3*128:])
        x=cv2.resize(np.mean(x, axis=-1), (64, 64))
        hog_features = hog(
            x,
            orientations=9,  # Number of gradient orientations
            pixels_per_cell=(8, 8),  # Size of each cell
            cells_per_block=(2, 2),  # Blocks of cells for normalization
            block_norm='L2-Hys',  # Normalization method
            # visualize=visualize,  # Return HOG image if True
            transform_sqrt=True  # Improve contrast
        )
        features.append(hog_features)
    return features
def extract_readings(file_path):
    """
    Extract readings from an EDF or BDF file based on the provided channel mapping.
    
    Parameters:
    - file_path (str): Path to the EDF or BDF file.
    
    Returns:
    - dict: Dictionary with channel names as keys and signal data as values.
    - dict: Dictionary with sampling rates for each channel.
    - dict: Dictionary with units for each channel.
    """
    # Channel mapping based on provided table
    channel_map = {
        1: 'Fp1', 2: 'AF3', 3: 'F3', 4: 'F7', 5: 'FC5', 6: 'FC1', 7: 'C3', 8: 'T7',
        9: 'CP5', 10: 'CP1', 11: 'P3', 12: 'P7', 13: 'PO3', 14: 'O1', 15: 'Oz', 16: 'Pz',
        17: 'Fp2', 18: 'AF4', 19: 'Fz', 20: 'F4', 21: 'F8', 22: 'FC6', 23: 'FC2', 24: 'Cz',
        25: 'C4', 26: 'T8', 27: 'CP6', 28: 'CP2', 29: 'P4', 30: 'P8', 31: 'PO4', 32: 'O2'
    }
    
    try:
        # Open the EDF/BDF file
        f = pyedflib.EdfReader(file_path)
        
        # Initialize dictionaries to store signals, sampling rates, and units
        readings = {}
        sampling_rates = {}
        units = {}
        
        # Get number of channels
        n_channels = f.signals_in_file
        print(f"Number of channels in file: {n_channels}")
        # Loop through each channel
        for i in range(n_channels):
            # Get channel label and clean it
            label = f.getLabel(i).strip()
            
            # Match channel label to the provided mapping (case-insensitive)
            channel_name = None
            for ch_num, ch_name in channel_map.items():
                if label.lower() == ch_name.lower():
                    print(f"Matched channel {label} to {ch_name}")
                    channel_name = ch_name
                    break
            
            # If channel is recognized, extract data
            if channel_name:
                # Read the signal data
                signal = f.readSignal(i)
                readings[channel_name] = signal
                
                # Get sampling rate
                sampling_rates[channel_name] = f.getSampleFrequency(i)
                
                # Get physical dimension (unit)
                units[channel_name] = f.getPhysicalDimension(i)
            else:
                print(f"Warning: Channel {label} not found in channel map.")
        
        # Close the file
        f.close()
        
        return readings, sampling_rates, units
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None


def preprocess_eeg_readings(signals, sampling_rates, units, original_fs=128):
    """
    Preprocess EEG readings for the first 32 channels based on specified steps.

    Parameters:
    - signals (dict): Dictionary of channel names and their signal data.
    - sampling_rates (dict): Dictionary of channel names and their sampling rates.
    - units (dict): Dictionary of channel names and their units.
    - original_fs (float): Original sampling frequency in Hz (default: 128).

    Returns:
    - dict: Preprocessed signals for the first 32 EEG channels.
    - dict: Updated sampling rates.
    - dict: Updated units.
    """
    # Define the Geneva order for the first 32 EEG channels
    geneva_order = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
        'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    ]
    # Initialize output dictionaries
    preprocessed_signals = {}
    updated_sampling_rates = {}
    updated_units = {}
    try:
        # Step 1: Process only the first 32 EEG channels
        eeg_channels = [ch for ch in geneva_order if ch in signals]
        
        # Step 2: Common average reference (CAR)
        # Compute the mean across all EEG channels for each time point
        eeg_data = np.array([signals[ch] for ch in eeg_channels])
        common_reference = np.mean(eeg_data, axis=0)
        
        for ch in eeg_channels:
            # Get original signal and sampling rate
            data = signals[ch]
            fs = sampling_rates[ch]
            
            # Step 3: Downsample to 128 Hz if original sampling rate is higher
            if fs > 128:
                num_samples = int(len(data) * 128 / fs)
                data = resample(data, num_samples)
                fs = 128
            
            # Step 4: Remove EOG artefacts (simplified approach based on regression)
            # Note: Full EOG removal as in [1] requires specific methodology (e.g., ICA or regression).
            # Here, we subtract a scaled version of the common reference.
            
            # Step 5: Apply bandpass filter (4.0-45.0 Hz)
            nyquist = fs / 2
            low = 4.0 / nyquist
            high = 45.0 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_data = filtfilt(b, a, data)
            
            # Step 6: Apply common average reference
            filtered_data = filtered_data - common_reference[:len(filtered_data)]
            
            # Store preprocessed signal
            preprocessed_signals[ch] = filtered_data
            updated_sampling_rates[ch] = fs
            updated_units[ch] = units[ch]
        
        # Step 7: Reorder channels to follow Geneva order
        ordered_signals = {ch: preprocessed_signals.get(ch, np.array([])) for ch in geneva_order}
        
        return ordered_signals, updated_sampling_rates, updated_units
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None, None
def extract_channel_readings (signals, valid_channels):
    """
    Extract readings from the specified channels in the signals dictionary.
    Parameters:
    - signals (dict): Dictionary containing EEG signals for various channels.
    - valid_channels (list): List of channel names to extract readings from.
    Returns:
    - np.ndarray: 2D array containing readings from the specified channels.
    """
    arr = np.vstack([signals[ch] for ch in valid_channels if signals[ch].size > 0])
    print("Channels used:", valid_channels)
    print("Shape of extracted readings:", arr.shape)
    return arr
def final_extraction(arr):
    """
    Perform final feature extraction on the EEG data.
    This function applies PCA and standard scaling to the EEG features.
    Parameters:
    - arr (np.ndarray): 2D array of EEG features.
    Returns:
    - np.ndarray: Transformed features after PCA and scaling.
    """
    pca = PCA(n_components=200)
    ss= StandardScaler()
    sig=Extract_EEG_Features3(arr)
    sig = np.array(sig)

    # Load the .pkl file
    pca_model=r"Models\EEG_models\EEG_pca.pkl"
    with open(pca_model, 'rb') as f:
        data = pickle.load(f)
    last_length = len(data)
    f_data=np.concatenate((data, sig), axis=0)
    f_data = pca.fit_transform(ss.fit_transform(f_data))
    data = f_data[last_length:]
    return data
    
def extract_trans_matrix(sequence):
    """
    Extract the transition matrix from a sequence of states (0 or 1).
    Parameters:
        sequence (list): List of states (0 or 1) representing the sequence.
    Returns:
        transition_matrix (np.ndarray): Transition matrix showing probabilities of moving from one state to another.
    """
    transitions = np.zeros((2, 2))  # states are 0 and 1

    # Count transitions
    for (current_state, next_state) in zip(sequence[:-1], sequence[1:]):
        transitions[current_state][next_state] += 1

    # Convert to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_matrix = transitions / row_sums  # Broadcasting handles division

    # Handle division by zero (if any state was never visited)
    transition_matrix = np.nan_to_num(transition_matrix)
    transition_matrix = np.transpose(transition_matrix)  # Transpose to match the expected format
    return transition_matrix
def simulate_markov_chain(transition_matrix, initial_distribution=np.array([[0.5],[0.5]]), tol=1e-8):
    """
    Simulate a Markov Chain and estimate the convergence to steady state.

    Parameters:
        transition_matrix (np.ndarray): The state transition matrix (NxN).
        initial_distribution (np.ndarray): Initial state distribution (Nx1).
        tol (float): Convergence tolerance.
    
    Returns:
        steady_state (np.ndarray): Estimated steady-state distribution.
        num_iterations (int): Number of iterations to convergence.
        history (list): List of state distributions at each step.
    """
    current_dist = initial_distribution.copy()
    history = [current_dist]
    next_dist=np.dot(transition_matrix,current_dist)
    history.append(next_dist)
    
    i=1
    # Check convergence
    if np.linalg.norm(next_dist - current_dist, ord=1) < tol:
        print(next_dist,current_dist,next_dist - current_dist)
        return next_dist
    
    current_dist = next_dist
    while np.linalg.norm(next_dist - current_dist, ord=1) < tol :
        next_dist=np.dot(transition_matrix,current_dist)
        
        # Check convergence
        if np.linalg.norm(next_dist - current_dist, ord=1) < tol:
            return next_dist
        current_dist = next_dist
def classifiy(data):
    """
    Classify the extracted features using a pre-trained model.
    Parameters:
    - data (np.ndarray): Extracted features from EEG signals.
    - model_path (str): Path to the pre-trained classification model.
    Returns:
    - predictions (list): List of predicted states (0 or 1) for each sample
    """
    model_path = 'Models/EEG_models/EEG_ET_valence_model.pkl'
    model = joblib.load(model_path)
    predictions = model.predict(data)
    return predictions
def extract_trans_matrix(sequence):
    """
    Extract the transition matrix from a sequence of states (0 or 1).
    Parameters:
        sequence (list): List of states (0 or 1) representing the sequence.
    Returns:
        transition_matrix (np.ndarray): Transition matrix showing probabilities of moving from one state to another.
    """
    transitions = np.zeros((2, 2))  # states are 0 and 1

    # Count transitions
    for (current_state, next_state) in zip(sequence[:-1], sequence[1:]):
        transitions[current_state][next_state] += 1

    # Convert to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_matrix = transitions / row_sums  # Broadcasting handles division

    # Handle division by zero (if any state was never visited)
    transition_matrix = np.nan_to_num(transition_matrix)
    transition_matrix = np.transpose(transition_matrix)  # Transpose to match the expected format
    return transition_matrix
def simulate_markov_chain(transition_matrix, initial_distribution=np.array([[0.5],[0.5]]), tol=1e-8):
    """
    Simulate a Markov Chain and estimate the convergence to steady state.

    Parameters:
        transition_matrix (np.ndarray): The state transition matrix (NxN).
        initial_distribution (np.ndarray): Initial state distribution (Nx1).
        tol (float): Convergence tolerance.
    
    Returns:
        steady_state (np.ndarray): Estimated steady-state distribution.
        num_iterations (int): Number of iterations to convergence.
        history (list): List of state distributions at each step.
    """
    current_dist = initial_distribution.copy()
    history = [current_dist]
    next_dist=np.dot(transition_matrix,current_dist)
    history.append(next_dist)
    
    i=1
    # Check convergence
    if np.linalg.norm(next_dist - current_dist, ord=1) < tol:
        print(next_dist,current_dist,next_dist - current_dist)
        return next_dist
    
    current_dist = next_dist
    while np.linalg.norm(next_dist - current_dist, ord=1) < tol :
        next_dist=np.dot(transition_matrix,current_dist)
        
        # Check convergence
        if np.linalg.norm(next_dist - current_dist, ord=1) < tol:
            return next_dist
        current_dist = next_dist
def final_prediction(file_path):
    """
    Extract features from an EEG file and classify the data using a pre-trained model.
    Parameters:
    - file_path (str): Path to the EEG file.
    - model_path (str): Path to the pre-trained classification model.
    Returns:
    - steady_state (np.ndarray): Estimated steady-state distribution from the Markov chain.
    - positive_states (np.ndarray): Steady state for positive states.
    - negative_states (np.ndarray): Steady state for negative states.
    """
    signals, sampling_rates, units = extract_readings(file_path)
    signals, rates, units = preprocess_eeg_readings(signals, sampling_rates, units)
    
    arr = extract_channel_readings(signals, geneva_order)
    data = final_extraction(arr)
    # print("Shape of final extracted data:", data.shape)
    # print("Number of samples:", data.shape[0])
    predictions = classifiy(data)
    # print("Predictions:", predictions)
    transition_matrix = extract_trans_matrix(predictions)
    # print("Transition Matrix:", transition_matrix)
    steady_state = simulate_markov_chain(transition_matrix)
    positive_states = steady_state[0][0]
    negative_states = steady_state[1][0]
    
    return steady_state , positive_states, negative_states