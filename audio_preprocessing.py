import os
import numpy as np
import pandas as pd
import joblib
import soundfile as sf
import opensmile
import librosa
from concurrent.futures import ThreadPoolExecutor

# استخراج egemaps من الصوت باستخدام opensmile

def extract_egemaps(segment_path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return smile.process_file(segment_path).reset_index(drop=True)

# استخراج mfcc من الصوت باستخدام librosa (حسب المطلوب للفيميل فقط - variance)
def extract_mfcc_variance(segment_path):
    y, sr = librosa.load(segment_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_var = np.var(mfcc, axis=1)
    delta_var = np.var(delta, axis=1)
    delta2_var = np.var(delta2, axis=1)

    all_features = {}
    for i in range(13):
        all_features[f'pcm_fftMag_mfcc[{i}]'] = [mfcc_var[i]]
        all_features[f'pcm_fftMag_mfcc_de[{i}]'] = [delta_var[i]]
        all_features[f'pcm_fftMag_mfcc_de_de[{i}]'] = [delta2_var[i]]

    return pd.DataFrame(all_features)

# تجهيز الفيتشرز لكل جزء من الصوت

def process_segment(i, segment, sr, user_type):
    user_type = user_type.lower().strip()
    if user_type not in ["female", "male"]:
        raise ValueError(f"Invalid user_type: {user_type}")

    temp_path = f"temp_segment_{i}.wav"
    sf.write(temp_path, segment, sr)

    try:
        if user_type == "female":
            combined = extract_mfcc_variance(temp_path)
        else:
            combined = extract_egemaps(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return combined

# تقسيم الصوت واستخراج الفيتشرز

def extract_features(audio_path, user_type):
    user_type = user_type.lower().strip()
    if user_type not in ["female", "male"]:
        raise ValueError(f"Invalid user_type: {user_type}")

    y, sr = sf.read(audio_path)
    n_windows = 4 if user_type == "female" else 2
    window_length = len(y) // n_windows

    segments = [y[i * window_length:(i + 1) * window_length if i < n_windows - 1 else len(y)] for i in range(n_windows)]

    features_list = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_segment, i, segments[i], sr, user_type) for i in range(n_windows)]
        for future in futures:
            features_list.append(future.result())

    final_df = pd.concat(features_list, ignore_index=True)
    final_mean_features = final_df.mean().to_frame().T
    return final_mean_features

# اختيار أسماء الفيتشرز المطلوبة فقط

def prepare_feature_names(user_type, features_df):
    user_type = user_type.lower().strip()
    if user_type not in ["female", "male"]:
        raise ValueError(f"Invalid user_type: {user_type}")

    if user_type == "female":
        mfcc_feature_names = (
            [f"pcm_fftMag_mfcc[{i}]" for i in range(13)] +
            [f"pcm_fftMag_mfcc_de[{i}]" for i in range(13)] +
            [f"pcm_fftMag_mfcc_de_de[{i}]" for i in range(13)]
        )
        expected_feature_names = mfcc_feature_names
    else:
        base_feature_names = [
            'loudness_sma3', 'alphaRatioV_sma3nz', 'hammarbergIndexV_sma3nz',
            'slopeV0-500_sma3nz', 'slopeV500-1500_sma3nz', 'spectralflux_sma3',
            'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3',
            'f0semitonefrom27.5hz_sma3nz', 'jitterlocal_sma3nz', 'shimmerlocaldb_sma3nz',
            'hnrdbacf_sma3nz', 'logrelf0-h1-h2_sma3nz', 'logrelf0-h1-a3_sma3nz',
            'f1frequency_sma3nz', 'f1bandwidth_sma3nz', 'f1amplitudelogrelf0_sma3nz',
            'f2frequency_sma3nz', 'f2amplitudelogrelf0_sma3nz', 'f3frequency_sma3nz',
            'f3amplitudelogrelf0_sma3nz'
        ]
        expected_feature_names = [f"{f}_amean" for f in base_feature_names]

    cols_lower = [c.lower() for c in features_df.columns]
    missing = [name for name in expected_feature_names if name.lower() not in cols_lower]

    if missing:
        print("\nMissing columns in extracted features:")
        for col in missing:
            print("-", col)
        raise ValueError("Some features are not available.")

    final_cols = []
    for name in expected_feature_names:
        for c in features_df.columns:
            if c.lower() == name.lower():
                final_cols.append(c)
                break

    return final_cols

# التنبؤ النهائي باستخدام الموديل

def audio_prediction(audio_path, user_type):
    user_type = user_type.lower().strip()
    if user_type not in ["female", "male"]:
        raise ValueError(f"Invalid user_type: {user_type}")

    features = extract_features(audio_path, user_type)
    feature_names = prepare_feature_names(user_type, features)
    features = features[feature_names]

    model_path = f"Models/Audio_models/{user_type}_model.pkl"
    scaler_path = f"Models/Audio_models/{user_type}_scaler.pkl"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(features)
    pred = model.predict(X_scaled)[0]
    proba_positive = model.predict_proba(X_scaled)[0][1]
    proba_negative = model.predict_proba(X_scaled)[0][0]

    return pred, proba_positive, proba_negative
