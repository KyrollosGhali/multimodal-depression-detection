# Audio-Based Depression Detection using Extended DIAC-Woz Dataset

## Overview
This notebook (`final Audio.ipynb`) implements a comprehensive audio-based depression detection system using the Extended DIAC-Woz dataset. The system employs advanced audio feature extraction techniques combined with machine learning to classify depression states based on speech patterns, with gender-specific model optimization.

## Key Features
- **Extended DIAC-Woz Dataset**: Comprehensive audio database for depression analysis
- **Gender-Specific Analysis**: Separate models optimized for male and female participants
- **Multi-Feature Extraction**: eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) and MFCC features
- **Advanced Preprocessing**: Variance-based feature processing and data segmentation
- **Class Imbalance Handling**: BorderlineSMOTE for balanced training
- **Multiple ML Models**: Random Forest (females) and Multi-Layer Perceptron (males)
- **Statistical Validation**: Chi-square tests and ROC analysis for model validation

## Dataset Information
- **Source**: Extended DIAC-Woz Dataset (Distress Interviews Analysis Corpus - Wizard of Oz)
- **Dataset Link**: [DIAC-Woz Database](https://dcapswoz.ict.usc.edu/daic-woz-database-download/)
- **Features**: 
  - **eGeMAPS**: 88 acoustic parameters including fundamental frequency, spectral, and prosodic features
  - **MFCC**: Mel-Frequency Cepstral Coefficients for speech representation
- **Labels**: Depression vs Non-Depression classification
- **Gender Split**: Separate analysis for male and female participants
- **Format**: OpenSMILE-extracted features in CSV format

## Methodology
1. **Data Organization**: Gender-based file separation and organization
2. **Feature Processing**: Variance-based segmentation and feature extraction
3. **Data Preprocessing**: Normalization, train-test split, and SMOTE balancing
4. **Model Selection**: Gender-specific model optimization (RF for females, MLP for males)
5. **Hyperparameter Tuning**: Grid search with cross-validation
6. **Evaluation**: Comprehensive performance assessment with statistical significance testing
7. **Model Persistence**: Save trained models for deployment

## Technical Stack
- **Audio Processing**: OpenSMILE feature extraction
- **Machine Learning**: Scikit-learn, XGBoost, TensorFlow
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Imbalanced Learning**: imbalanced-learn (BorderlineSMOTE)
- **Statistical Analysis**: SciPy for significance testing

## Requirements
```
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
imbalanced-learn
scipy
xgboost
```

## Usage
1. Ensure the Extended DIAC-Woz dataset is downloaded and properly organized
2. Run the notebook cells sequentially to reproduce the analysis
3. The notebook will generate gender-specific models and evaluation metrics
4. Trained models are saved for future use

## Results
The system achieves optimal performance through gender-specific model selection:
- **Female participants**: Random Forest classifier
- **Male participants**: Multi-Layer Perceptron
- Comprehensive evaluation includes ROC-AUC, confusion matrices, and statistical significance testing

## File Structure
- `final Audio.ipynb`: Main notebook with complete analysis pipeline
- Supporting data files (not included in repository)
- Generated model files (saved after training)
