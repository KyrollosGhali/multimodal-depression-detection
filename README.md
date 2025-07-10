# MultiModal Depression Detection (MMDD)

A comprehensive web application for depression detection using multiple modalities including text analysis, video processing, audio analysis, and EEG signals. The system provides an interactive chatbot interface with multilingual support and real-time analysis capabilities.

## � Demo Video

**Watch the MMDD System in Action!**

📹 **Demo Video**: [`video_2025-07-07_00-34-41.mp4`](./video_2025-07-07_00-34-41.mp4)

This comprehensive demonstration video showcases:
- **Complete User Workflow**: From initial setup to final results
- **Multimodal Analysis**: All four analysis modalities working together
- **Real-time Processing**: Live video and audio capture during chat sessions
- **Interactive Chatbot**: AI-powered conversation with emotional support
- **Multilingual Interface**: Language selection and TTS capabilities
- **Results Visualization**: How depression risk scores are presented
- **User Experience**: Intuitive navigation through all application features

> **Note**: The demo video provides a complete walkthrough of the system's capabilities and is highly recommended for understanding the full potential of the MMDD application.

## �🌟 Features

### Core Functionality
- **Multimodal Analysis**: Combines text, video, audio, and EEG data for comprehensive depression detection
- **Interactive Chatbot**: AI-powered conversational interface with emotional support
- **Multilingual Support**: Available in English, Arabic, French, and German
- **Real-time Processing**: Live video recording and audio capture during chat sessions
- **Gender-specific Models**: Optimized models for male and female users
- **Text-to-Speech**: Natural voice responses in multiple languages

### Analysis Modalities

#### 1. Text Analysis
- **Technology**: Sentence Transformers + Deep Learning
- **Features**: Advanced embeddings using 'all-mpnet-base-v2'
- **Model**: Neural network with regularization techniques
- **Capability**: Analyzes conversation transcripts for depression indicators

#### 2. Video Processing
- **Technology**: MediaPipe Face Mesh + Computer Vision
- **Features**: 468 facial landmarks with 3D coordinates (1,404 features)
- **Model**: SVM classifier for facial expression analysis
- **Capability**: Real-time facial emotion recognition during conversations

#### 3. Audio Analysis
- **Technology**: OpenSMILE + Gender-specific ML models
- **Features**: 
  - eGeMAPS (88 parameters) for male users
  - MFCC variance features for female users
- **Models**: Random Forest (females), Multi-Layer Perceptron (males)
- **Capability**: Voice pattern analysis for emotional state detection

#### 4. EEG Signal Processing
- **Technology**: EFDM (Enhanced Frequency Domain Mapping) + HOG features
- **Features**: Spatial-spectral representations from 32 EEG channels
- **Model**: Random Forest with PCA dimensionality reduction
- **Capability**: Brain activity pattern analysis for emotional state assessment

## 🚀 Getting Started

### Prerequisites
- Python 3.9.13 (see requirements.txt for version details)
- Web browser with camera and microphone access
- Optional: EEG device compatible with EDF format

> 💡 **Quick Start Tip**: Watch the demo video (`video_2025-07-07_00-34-41.mp4`) first to understand the complete workflow before installation.

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MMDD
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   - Update the Groq API key in `chatbot.py` (line 30 and 125)
   - Replace with your own API key for production use

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your web browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
MMDD/
├── app.py                      # Main Flask application
├── chatbot.py                  # AI chatbot with multilingual support
├── video_processing.py         # Facial landmark extraction and analysis
├── audio_preprocessing.py      # Audio feature extraction and prediction
├── EEG_preprocessing.py        # EEG signal processing and analysis
├── text_preprocessing.py       # Text embedding and sentiment analysis
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── video_2025-07-07_00-34-41.mp4  # Demo video showing system functionality
├── flask-71cd5-firebase-adminsdk-fbsvc-a66f40ca13.json  # Firebase config
├── Models/                     # Pre-trained ML models
│   ├── Audio_models/          # Gender-specific audio models
│   ├── EEG_models/           # EEG classification models
│   ├── text_model/           # Text analysis model
│   └── vision_model/         # Facial expression model
├── static/                    # Web assets
│   ├── CSS/                  # Stylesheets
│   ├── JS/                   # JavaScript files
│   ├── uploads/              # User uploaded files
│   ├── user_voices/          # Audio recordings
│   ├── videos/               # Video recordings
│   └── edfs/                 # EEG files
├── templates/                 # HTML templates
│   ├── home.html             # Landing page
│   ├── chat.html             # Chat interface
│   ├── upload_eeg.html       # EEG upload page
│   ├── processing.html       # Analysis progress page
│   ├── result.html           # Results display
│   └── splash.html           # Welcome screen
├── prompts/                   # Multilingual chatbot prompts
│   ├── en.txt                # English prompts
│   ├── ar.txt                # Arabic prompts
│   ├── fr.txt                # French prompts
│   └── de.txt                # German prompts
└── Final NoteBooks/           # Research notebooks and datasets
    ├── final Audio.ipynb     # Audio analysis development
    ├── Final EEG.ipynb       # EEG processing development
    ├── Final image file.ipynb # Video analysis development
    ├── Final Texture file.ipynb # Text analysis development
    └── depression_dataset.csv # Training dataset
```

## 🔧 Usage

### 1. Basic Session
1. Select your preferred language (English, Arabic, French, German)
2. Choose your gender for optimized model selection
3. Select voice preference for chatbot responses
4. Start chatting with the AI assistant

### 2. Complete Analysis
1. Complete a conversation session with the chatbot
2. The system automatically records video during the session
3. Upload an EEG file (optional, .edf format)
4. Wait for processing to complete
5. View comprehensive results with depression risk assessment

### 3. Supported File Formats
- **Video**: Automatically recorded via web browser (WebM format)
- **Audio**: Automatically captured during conversation (WAV format)
- **EEG**: EDF files from compatible EEG devices
- **Text**: Conversation transcripts (automatically generated)

## 🤖 AI Models

### Text Analysis Model
- **Architecture**: Deep Neural Network
- **Input**: Sentence embeddings (1536 dimensions)
- **Output**: Depression probability score
- **Training**: SMOTE-balanced dataset with early stopping

### Video Analysis Model
- **Architecture**: Support Vector Machine
- **Input**: 468 facial landmarks (1,404 features)
- **Output**: Emotional state classification
- **Features**: MediaPipe Face Mesh 3D coordinates

### Audio Analysis Models
- **Female Model**: Random Forest with MFCC variance features
- **Male Model**: Multi-Layer Perceptron with eGeMAPS features
- **Input**: Gender-specific audio features
- **Output**: Depression risk probability

### EEG Analysis Model
- **Architecture**: Random Forest with PCA
- **Input**: HOG features from EFDM representations
- **Output**: Valence-based emotion classification
- **Processing**: 32-channel EEG with STFT transformation

## 🌐 Multilingual Support

The application supports four languages with full localization:

- **English (en)**: Default language with comprehensive support
- **Arabic (ar)**: Right-to-left text support with Egyptian dialect conversion
- **French (fr)**: Complete French language interface
- **German (de)**: Full German language support

### Language Features
- Automatic text translation for cross-language communication
- Language-specific TTS voices for natural responses
- Localized UI elements and prompts
- Cultural adaptation for Arabic users (Egyptian dialect)

## 📊 Results Interpretation

The system provides a comprehensive depression risk score (0-100%) based on:

- **Text Analysis**: Conversation sentiment and emotional indicators
- **Video Analysis**: Facial expression patterns and micro-expressions
- **Audio Analysis**: Voice pattern analysis and prosodic features
- **EEG Analysis**: Brain activity patterns (if available)

Results are presented with:
- Overall risk score with visual progress indicator
- Individual modality scores
- Confidence levels for each analysis
- Recommendations based on risk level

## 🔒 Privacy and Security

- **Local Processing**: All analysis performed locally on your machine
- **Session-based**: Data automatically cleared after each session
- **No Data Storage**: User data not permanently stored
- **Secure API**: External API calls limited to translation and TTS services

## 🛠️ Development

### Adding New Languages
1. Create prompt file in `prompts/{language_code}.txt`
2. Add TTS voice mapping in `app.py` and `chatbot.py`
3. Update UI translations in HTML templates

### Model Updates
1. Replace model files in respective `Models/` subdirectories
2. Update feature extraction functions if needed
3. Test with sample data before deployment

### API Configuration
- Update Groq API key in `chatbot.py`
- Configure Firebase settings if using cloud features
- Modify model paths in preprocessing modules

## 📝 Research Background

This project is based on extensive research in multimodal depression detection:

- **Datasets**: DEAP (EEG), Extended DIAC-Woz (Audio), Custom datasets (Text, Video)
- **Techniques**: SMOTE, Cross-validation, Hyperparameter tuning
- **Validation**: Statistical significance testing, K-fold cross-validation
- **Performance**: Optimized for real-world deployment scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for research and educational purposes. Please ensure compliance with local regulations when using for clinical applications.

## ⚠️ Disclaimer

This application is designed for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions regarding mental health conditions.

## 📞 Support

For technical support or questions about the research:
- **📹 First**: Watch the demo video (`video_2025-07-07_00-34-41.mp4`) for a complete system overview
- Check the documentation in `Final NoteBooks/README.md`
- Review the implementation notebooks for detailed methodology
- Contact the development team for specific technical issues

### Quick Troubleshooting
If you encounter issues:
1. Ensure Python 3.9.13 is installed
2. Verify all dependencies from `requirements.txt` are installed
3. Check that your browser allows camera and microphone access
4. Refer to the demo video for correct usage patterns

---

**Note**: Remember to update API keys and configure paths according to your environment before deployment.
