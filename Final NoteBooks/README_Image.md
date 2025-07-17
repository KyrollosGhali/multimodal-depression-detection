# Facial Expression Recognition using MediaPipe and Machine Learning

## Overview
This notebook (`Final image file.ipynb`) implements a comprehensive facial expression recognition system that classifies emotions (happy vs sad) using facial landmarks extracted from images. The project employs multiple machine learning approaches including deep neural networks, ensemble methods, and support vector machines.

## Key Features
- **MediaPipe Face Mesh**: Advanced facial landmark extraction (468 landmarks per face)
- **Multiple ML Approaches**: Deep Neural Networks, Random Forest, Extra Trees, SVM
- **Hyperparameter Tuning**: Comprehensive model optimization
- **Class Imbalance Handling**: SMOTE for balanced training
- **Comprehensive Evaluation**: Cross-validation, confusion matrices, ROC analysis
- **Real-time Processing**: Efficient landmark extraction and classification

## Dataset Information
- **Source**: Affect Net Happy Sad Dataset
- **Dataset Link**: [Affect Net Happy Sad](https://www.kaggle.com/datasets/eldarsharapov/affect-net-happy-sad)
- **Classes**: 
  - **Happy**: Positive emotion expressions
  - **Sad**: Negative emotion expressions
- **Format**: Facial images with emotion labels

## Methodology
1. **Feature Extraction**: MediaPipe Face Mesh for 468 3D facial landmarks
2. **Data Preprocessing**: Train-test split, data shuffling, SMOTE oversampling
3. **Model Training**: 
   - Deep Neural Networks with hyperparameter tuning
   - Random Forest and Extra Trees ensemble methods
   - Support Vector Machines with different kernels
4. **Evaluation**: Accuracy, Confusion Matrix, ROC-AUC, Cross-validation
5. **Model Comparison**: Performance assessment across different algorithms

## Technical Stack
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: Scikit-learn, TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Imbalanced Learning**: imbalanced-learn (SMOTE)

## MediaPipe Face Mesh
The notebook utilizes MediaPipe's state-of-the-art face mesh detection:
- **468 3D facial landmarks** per detected face
- Real-time processing capabilities
- Robust landmark detection across various lighting conditions
- High-precision facial geometry analysis

## Model Architecture
### Deep Neural Network
- Multi-layer perceptron with dropout regularization
- Hyperparameter tuning for optimal performance
- Cross-validation for robust evaluation

### Ensemble Methods
- Random Forest classifier
- Extra Trees classifier
- Feature importance analysis

### Support Vector Machines
- Multiple kernel options (RBF, linear, polynomial)
- Hyperparameter optimization
- Cross-validation assessment

## Requirements
```
numpy
pandas
matplotlib
seaborn
opencv-python
mediapipe
scikit-learn
tensorflow
imbalanced-learn
```

## Usage
1. Download the Affect Net Happy Sad dataset
2. Organize images in the expected directory structure
3. Run the notebook cells sequentially to reproduce the analysis
4. The notebook will extract facial landmarks and train multiple models
5. Comparative evaluation results will be generated

## Results
The system achieves emotion recognition through:
- Precise facial landmark extraction using MediaPipe
- Comprehensive model comparison across different ML approaches
- Robust evaluation with cross-validation and statistical metrics
- Detailed performance analysis and visualization

## File Structure
- `Final image file.ipynb`: Main notebook with complete facial expression analysis
- Supporting image datasets (not included in repository)
- Generated model files and evaluation results

## Innovation
This notebook demonstrates:
- Integration of MediaPipe's advanced computer vision capabilities
- Comprehensive comparison of machine learning approaches
- Robust evaluation methodology for facial expression recognition
- Real-time processing potential for emotion detection applications
