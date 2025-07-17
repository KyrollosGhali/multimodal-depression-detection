# EEG-Based Emotion Recognition using EFDM and Machine Learning

## Overview
This notebook (`Final EEG.ipynb`) implements a comprehensive EEG (Electroencephalography) based emotion recognition system using the DEAP dataset. The project transforms EEG signals into Enhanced Frequency Domain Maps (EFDM) and applies machine learning techniques to classify emotional states (positive vs negative valence).

## Key Features
- **DEAP Dataset Processing**: Load and preprocess EEG signals from multiple subjects
- **STFT Analysis**: Short-Time Fourier Transform for time-frequency decomposition
- **EFDM Generation**: Enhanced Frequency Domain Mapping for spatial-spectral representation
- **HOG Feature Extraction**: Histogram of Oriented Gradients from EFDM images
- **Dimensionality Reduction**: PCA and StandardScaler for feature optimization
- **Multiple ML Models**: Random Forest, Extra Trees, and LightGBM classifiers
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, and ROC analysis

## Dataset Information
- **Source**: DEAP Dataset (Database for Emotion Analysis using Physiological Signals)
- **Dataset Link**: [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html)
- **Channels**: 32 EEG electrodes
- **Sampling Rate**: 128 Hz
- **Labels**: Valence (positive/negative emotion) based on self-assessment ratings
- **Trials**: Multiple video-watching sessions per subject

## Methodology
1. **Data Loading**: Load preprocessed DEAP EEG data files
2. **Signal Processing**: Apply STFT to extract time-frequency information
3. **EFDM Creation**: Map electrode positions to spatial grid with frequency content
4. **Feature Engineering**: Extract HOG features from EFDM representations
5. **Data Preprocessing**: Apply SMOTE, scaling, and PCA
6. **Model Training**: Train and tune multiple machine learning models
7. **Evaluation**: Comprehensive performance assessment and visualization

## Technical Stack
- **Signal Processing**: SciPy, NumPy
- **Image Processing**: OpenCV, scikit-image (HOG)
- **Machine Learning**: Scikit-learn, LightGBM
- **Data Handling**: Pandas, Pickle
- **Visualization**: Matplotlib, Seaborn
- **Imbalanced Learning**: imbalanced-learn (SMOTE)

## Enhanced Frequency Domain Maps (EFDM)
The notebook introduces a novel approach to EEG analysis by:
- Converting EEG signals to time-frequency representations using STFT
- Mapping electrode positions to spatial locations
- Creating 2D frequency domain maps for each time window
- Extracting HOG features from these spatial-spectral representations

## Requirements
```
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn
scikit-image
opencv-python
lightgbm
imbalanced-learn
pickle
```

## Usage
1. Download and preprocess the DEAP dataset
2. Organize EEG data files in the expected format
3. Run the notebook cells sequentially to reproduce the analysis
4. The notebook will generate EFDM representations and train classification models
5. Evaluation metrics and visualizations will be produced

## Results
The system achieves emotion recognition through:
- Novel EFDM representation of EEG signals
- HOG feature extraction from spatial-spectral maps
- Multiple classifier comparison (Random Forest, Extra Trees, LightGBM)
- Comprehensive evaluation with cross-validation and ROC analysis

## File Structure
- `Final EEG.ipynb`: Main notebook with complete EFDM-based analysis
- Supporting EEG data files (not included in repository)
- Generated EFDM maps and feature files

## Innovation
This notebook presents an innovative approach to EEG-based emotion recognition by:
- Transforming temporal EEG signals into spatial-spectral representations
- Utilizing computer vision techniques (HOG) for feature extraction
- Bridging the gap between signal processing and image analysis in neuroscience
