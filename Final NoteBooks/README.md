# Mental Health Detection using Multimodal Machine Learning

This repository contains a comprehensive suite of machine learning models for mental health detection using four different modalities: **facial expressions**, **text analysis**, **EEG signals**, and **audio analysis**. Each approach employs state-of-the-art techniques in their respective domains.

## 📁 Repository Structure

```
Final NoteBooks/
├── Final image file.ipynb      # Facial Expression Recognition
├── Final Texture file.ipynb    # Text-based Depression Detection  
├── Final EEG.ipynb             # EEG-based Emotion Recognition
├── final Audio.ipynb           # Audio-based Depression Detection
└── README.md                   # This file
```

## 🎯 Project Overview

### 1. Facial Expression Recognition (`Final image file.ipynb`)
- **Technology**: MediaPipe Face Mesh + Deep Learning
- **Features**: 468 facial landmarks with 3D coordinates (1,404 features)
- **Models**: Neural Networks, Random Forest, Extra Trees, SVM
- **Dataset**: Happy vs Sad facial images
- **Key Techniques**: SMOTE, Cross-validation, Hyperparameter tuning

**📊 Key Visualizations:**
- **Training/Testing Data Distribution**: Side-by-side bar charts showing balanced distribution of ~4,300 happy and sad samples in training, and ~1,000 samples each in testing
- **K-Fold Cross-Validation Results**: Bar chart displaying validation accuracy across 5 folds (ranging from 0.50 to 0.87), with Fold 4 achieving the highest performance
- **Confusion Matrix**: Heatmap showing model performance with 898 true positives for happy, 909 true positives for sad, and relatively low false positives (181 and 128)
- **Training History**: Dual-panel plot showing model accuracy and loss curves over 100 epochs, demonstrating convergence around epoch 40 with final accuracy ~85%
- **ROC Curve**: Performance curve achieving AUC = 0.93, indicating excellent discriminative ability between happy and sad expressions


### 2. Text-based Depression Detection (`Final Texture file.ipynb`)
- **Technology**: Sentence Transformers + NLP
- **Features**: Advanced text embeddings using 'all-mpnet-base-v2'
- **Models**: Deep Neural Networks with regularization
- **Dataset**: Text paragraphs labeled as Normal vs Depression
- **Key Techniques**: Mean/Max pooling, SMOTE, Early stopping

**📊 Key Visualizations:**
- **Mental Health Class Distribution**: Bar chart showing perfectly balanced dataset with 5,000 samples each for Normal (0) and Depression (1) classes
- **Model Training Accuracy**: Learning curve demonstrating rapid convergence to near-perfect accuracy (1.00) within first few epochs, with both training and validation maintaining consistent performance
- **ROC Curve**: Perfect classification performance with AUC = 1.00, showing the model's ability to completely separate normal and depression text samples
- **Classification Report**: Perfect precision, recall, and F1-scores (1.00) for both classes, indicating exceptional model performance on text-based depression detection

### 3. EEG-based Emotion Recognition (`Final EEG.ipynb`)
- **Technology**: EFDM (Enhanced Frequency Domain Mapping) + HOG
- **Features**: Spatial-spectral representations from 32 EEG channels
- **Models**: Random Forest, Extra Trees, LightGBM
- **Dataset**: DEAP dataset (valence-based emotion classification)
- **Key Techniques**: STFT, PCA, Cross-validation

**📊 Key Visualizations:**
- **Aggregated STFT Representation**: Time-frequency heatmap (128×64) showing EEG activity across all channels, with frequency bins up to 64Hz and time progression up to 120 bins
- **EFDM Spatial Representation**: 4×8 electrode grid heatmap displaying averaged frequency power across spatial locations, with values ranging from 0.368 to 4.083
- **Processed EFDM Image**: 64×64 pixel representation of spatial-spectral brain activity, normalized for HOG feature extraction with distinct activation patterns
- **Label Distribution Charts**: 
  - Original imbalanced dataset: 712 positive vs 288 negative samples
  - Post-SMOTE balanced dataset: Equal distribution of positive and negative samples
- **Cross-Validation Performance**: Bar chart showing Extra Trees Classifier accuracy across 5 folds (ranging from 0.83 to 0.89)
- **Confusion Matrix**: Final model performance with 83% accuracy, showing 165 and 168 correct predictions with 36 and 33 misclassifications
- **ROC Curve**: Model discrimination ability with AUC = 0.89, demonstrating strong predictive performance for emotion recognition

### 4. Audio-based Depression Detection (`final Audio.ipynb`)
- **Technology**: OpenSMILE Feature Extraction + Gender-Specific ML
- **Features**: eGeMAPS (88 parameters) for males, MFCC for females
- **Models**: Random Forest (females), Multi-Layer Perceptron (males)
- **Dataset**: Extended DIAC-Woz dataset with gender-specific analysis
- **Key Techniques**: Variance-based segmentation, BorderlineSMOTE, Statistical validation

**📊 Key Visualizations:**

**Female Model Performance:**
- **Confusion Matrix**: Shows 80% accuracy with 20 true negatives, 25 true positives, 8 false positives, and 3 false negatives
- **ROC Curve**: Strong performance with AUC = 0.8801, indicating good discriminative ability
- **Cross-Validation**: 5-fold validation showing mean accuracy of 81.7% and F1-score of 81.8%
- **Statistical Significance**: Chi-square test (χ² = 18.89, p < 0.0001) confirming model performance is significantly better than random

**Male Model Performance:**
- **Confusion Matrix**: Shows 82% accuracy with 10 true negatives, 13 true positives, 4 false positives, and 1 false negative
- **ROC Curve**: Moderate performance with AUC = 0.6888
- **Cross-Validation**: 5-fold validation achieving mean accuracy of 85.7% and F1-score of 85.6%
- **Statistical Significance**: Chi-square test (χ² = 9.58, p = 0.002) confirming significant performance above chance level

**Gender-Specific Analysis**: The visualizations demonstrate that different feature extraction methods (eGeMAPS vs MFCC) and models (Random Forest vs MLP) are optimal for different genders, with females showing higher AUC but males achieving better cross-validation accuracy.

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow keras opencv-python mediapipe
pip install sentence-transformers nltk imbalanced-learn
pip install lightgbm scipy scikit-image joblib
pip install xgboost opensmile
```

### Running the Notebooks

1. **Update Dataset Paths**: Modify the file paths in each notebook to match your dataset locations
2. **Install Dependencies**: Run the import cells to ensure all libraries are available
3. **Execute Sequentially**: Run cells in order for proper functionality
4. **Monitor Progress**: Each notebook includes progress indicators for long-running processes

## 📊 Model Performance

### Facial Expression Recognition
- **Best Model**: Neural Network with optimized architecture
- **Accuracy**: 85% on test set with balanced happy/sad classification
- **Key Metrics**: AUC = 0.93, demonstrating excellent discriminative ability
- **Visualizations**: Training curves, confusion matrix, ROC analysis, K-fold validation charts

### Text Depression Detection  
- **Best Model**: Deep Neural Network with sentence embeddings
- **Accuracy**: Perfect 100% classification on normal vs depression text
- **Key Metrics**: AUC = 1.00, precision/recall = 1.00 for both classes
- **Visualizations**: Class distribution, learning curves, ROC curve, performance metrics

### EEG Emotion Recognition
- **Best Model**: Extra Trees Classifier with HOG features from EFDM
- **Accuracy**: 83% on DEAP dataset valence classification
- **Key Metrics**: AUC = 0.89, consistent cross-validation performance (83-89%)
- **Visualizations**: STFT heatmaps, spatial electrode maps, EFDM images, validation charts

### Audio Depression Detection
- **Best Model**: Gender-specific optimization (Random Forest for females, MLP for males)
- **Accuracy**: 80% (females), 82% (males) with statistical significance validation
- **Key Metrics**: Female AUC = 0.88, Male AUC = 0.69, both p < 0.05 significance
- **Visualizations**: Gender-specific confusion matrices, ROC curves, cross-validation results

## 🛠️ Technical Details

### Data Preprocessing
- **Facial**: MediaPipe landmark extraction, data augmentation with SMOTE
- **Text**: Tokenization, lemmatization, stopword removal, embedding generation
- **EEG**: STFT analysis, spatial mapping, image processing, HOG extraction
- **Audio**: OpenSMILE feature extraction, gender-based file organization, variance-based segmentation

### Feature Engineering
- **Facial**: 3D coordinate normalization, landmark relationships
- **Text**: Mean and max pooling of sentence embeddings
- **EEG**: Time-frequency analysis, spatial electrode mapping
- **Audio**: eGeMAPS and MFCC features, temporal variance analysis, gender-specific processing

### Model Optimization
- **Hyperparameter Tuning**: Grid search, cross-validation
- **Regularization**: L1/L2 penalties, dropout, early stopping
- **Class Balance**: SMOTE oversampling, stratified sampling

## 📈 Results and Visualizations

### Facial Expression Recognition Charts
- **Data Distribution**: Training/testing split visualization showing balanced happy/sad samples
- **Cross-Validation**: 5-fold accuracy performance across different validation sets
- **Confusion Matrix**: Detailed prediction accuracy heatmap with true/false positives
- **Training History**: Accuracy and loss curves over 100 training epochs
- **ROC Curve**: Receiver Operating Characteristic with AUC = 0.93

### Text Depression Detection Charts  
- **Class Distribution**: Balanced dataset visualization (5,000 Normal vs 5,000 Depression)
- **Training Accuracy**: Learning curve showing rapid convergence to perfect accuracy
- **ROC Analysis**: Perfect classification curve with AUC = 1.00
- **Performance Metrics**: Complete classification report with precision/recall scores

### EEG Emotion Recognition Charts
- **STFT Representation**: Time-frequency heatmap of aggregated EEG signals (128×64)
- **EFDM Spatial Map**: 4×8 electrode grid showing frequency power distribution
- **Processed Images**: 64×64 normalized brain activity representations for HOG extraction
- **Data Balance**: Before/after SMOTE visualization showing class distribution
- **Cross-Validation**: Extra Trees performance across 5 folds (83-89% accuracy)
- **Confusion Matrix**: Final model results with 83% overall accuracy
- **ROC Curve**: Discrimination performance with AUC = 0.89

### Audio Depression Detection Charts
- **Female Model Visualizations**:
  - Confusion matrix (80% accuracy, 56 total samples)
  - ROC curve with AUC = 0.8801
  - 5-fold cross-validation results (mean 81.7% accuracy)
- **Male Model Visualizations**:
  - Confusion matrix (82% accuracy, 28 total samples)  
  - ROC curve with AUC = 0.6888
  - 5-fold cross-validation results (mean 85.7% accuracy)
- **Statistical Validation**: Chi-square significance testing for both gender models

## 🔬 Research Applications

This work can be applied to:
- **Clinical Assessment**: Automated screening tools
- **Mental Health Monitoring**: Continuous assessment systems
- **Research**: Multimodal mental health studies
- **Healthcare Technology**: Integration into digital health platforms

## 📝 Usage Guidelines

### For Researchers
- Cite appropriate datasets (DEAP for EEG data)
- Follow ethical guidelines for mental health research
- Validate results on independent datasets

### For Developers
- Update file paths for your environment
- Adjust hyperparameters based on your hardware
- Consider computational requirements for large datasets

### For Practitioners
- Ensure proper data privacy and security
- Validate models before clinical use
- Consider multimodal fusion for improved accuracy

## ⚠️ Important Notes

- **Dataset Paths**: Update all file paths to match your system
- **Computational Resources**: Some processes are computationally intensive
- **Model Validation**: Results should be validated on independent datasets
- **Ethical Considerations**: Follow appropriate guidelines for mental health research

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create feature branches for improvements
3. Follow coding standards and documentation practices
4. Submit pull requests with detailed descriptions

## 📄 License

This project is intended for research and educational purposes. Please ensure compliance with dataset licenses and ethical guidelines.

## 📧 Contact

For questions, issues, or collaborations, please open an issue in the repository or contact the development team.

---

**Note**: This is a research project focusing on machine learning applications in mental health. Results should be validated and used responsibly in accordance with medical and ethical standards.
