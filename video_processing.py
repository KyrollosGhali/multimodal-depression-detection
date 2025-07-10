import numpy as np
import cv2
import mediapipe as mp
import pickle
import av
import wave
def read_video(video_path):
    """
    Process a video file to extract facial landmarks using MediaPipe FaceMesh.
    
    Parameters:
        video_path (str): Path to the input video file.
    
    Returns:
        features (np.ndarray): Array of extracted features from the video frames.
    """
# Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)

    # Your 328 selected indices (for example, use full range to test)
    selected_indices = list(range(468))  # Replace with specific 328 landmark indices if known

    # Open video
    cap = cv2.VideoCapture(video_path)

    all_features = []  # To store feature vectors from all frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            features = []
            for idx in selected_indices:
                lm = landmarks[idx]
                features.extend([lm.x, lm.y, lm.z])  # Add x, y, z

            if len(features) == 1404:
                all_features.append(features)
            else:
                print(f"Frame skipped: only got {len(features)} features")

    cap.release()

    # Convert to numpy array for saving or processing
    features = np.array(all_features)  # shape: (num_frames, 984)
    return features
def classifiy(features):
    """ Classify the extracted features using a pre-trained SVM model.
    Parameters:
        features (np.ndarray): Array of extracted features from the video frames.
    Returns:
        predictions (list): List of predicted states (0 or 1) for each frame.
    """
    model_path = r"Models\vision_model\svm_visual_model.pkl"
    model = pickle.load(open(model_path, 'rb'))
    predictions = [model.predict(np.array(feature).reshape(1, -1))[0] for feature in features]
    print(predictions)
    return predictions
def extract_trans_matrix(sequence):
    """ Extract the transition matrix from a sequence of states (0 or 1).
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
def simulate_markov_chain(transition_matrix, initial_distribution=np.array([[0.5],[0.5]]), tol=0.00000001):
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
    print(f'The iter number {i} :\n{next_dist}\n{current_dist}\n{next_dist - current_dist}\n{np.linalg.norm(next_dist - current_dist, ord=1)}')
    # Check convergence
    if np.linalg.norm(next_dist - current_dist, ord=1) < tol:
        print(next_dist,current_dist,next_dist - current_dist)
        return next_dist, i, history
    
    current_dist = next_dist
    while np.linalg.norm(next_dist - current_dist, ord=1) < tol:
        current_dist = next_dist
        next_dist=np.dot(transition_matrix,current_dist)
        history.append(next_dist)
        
        # Check convergence
        if np.linalg.norm(next_dist - current_dist, ord=1) < tol:
            return next_dist, i, history
        i+=1
        
        print(f'The iter number {i} :\n{next_dist}\n{current_dist}\n{next_dist - current_dist}\n{np.linalg.norm(next_dist - current_dist, ord=1)}')
        current_dist = next_dist
def visual_pred(video_path):
    """
    Process a video file to extract features
    , classify them, and simulate a Markov chain to get the final result.
    video_path (str): Path to the input video file.
    predictions (list): List of predicted states from the classification model.
    sequence (list): Sequence of states extracted from predictions.
    final_result (np.ndarray): Final result from the Markov chain simulation.
    positive_prob (float): Probability of being in the positive state.
    negative_prob (float): Probability of being in the negative state.
    """
    features=read_video(video_path)
    predictions=classifiy(features)
    sequance=extract_trans_matrix(predictions)
    final_result,_,_=simulate_markov_chain(sequance)
    positive_prob = final_result[0][0]
    negative_prob = final_result[1][0]
    return final_result , negative_prob, positive_prob
def video_to_audio(video_path, output_file):
    """
    Extract audio from a video file and save it as a WAV file.
    
    Parameters:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio in WAV format.
    """
    container = av.open(video_path)

    # Get audio stream
    stream = next(s for s in container.streams if s.type == 'audio')

    codec_ctx = stream.codec_context
    sample_rate = codec_ctx.sample_rate
    channels = codec_ctx.channels
    format_name = codec_ctx.format.name

    print(f"Format: {format_name}, Rate: {sample_rate}, Channels: {channels}")

    # Set sample width based on format
    if "s16" in format_name:
        sample_width = 2
        conversion = lambda f: f.to_ndarray()
    elif "s32" in format_name:
        sample_width = 4
        conversion = lambda f: f.to_ndarray()
    elif "flt" in format_name:
        sample_width = 2  # We'll convert to 16-bit PCM
        def conversion(f):
            # Convert float32 to int16 range
            arr = f.to_ndarray()
            arr = np.clip(arr, -1, 1)  # Normalize if needed
            return (arr * 32767).astype(np.int16)
    else:
        raise ValueError(f"Unsupported format: {format_name}")

    with wave.open(output_file, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)

        for frame in container.decode(stream):
            audio_data = conversion(frame)
            wav_file.writeframes(audio_data.tobytes())
