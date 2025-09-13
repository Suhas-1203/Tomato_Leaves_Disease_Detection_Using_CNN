# Tomato Leaf Disease Detection System

A comprehensive deep learning-based desktop application for automated detection and classification of tomato leaf diseases using Convolutional Neural Networks (CNN) and a user-friendly Tkinter GUI interface.

## 🌟 Project Overview

This project leverages deep learning to help farmers, agricultural researchers, and gardeners identify common tomato leaf diseases quickly and accurately. The system uses a custom-built CNN model trained on a comprehensive dataset of tomato leaf images to classify diseases and provide instant results through an intuitive graphical interface.

## 🚀 Features

- **Real-time Disease Detection**: Upload tomato leaf images and get instant disease predictions
- **10 Disease Classifications**: Detects Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, and Healthy leaves
- **User-Friendly GUI**: Clean, intuitive Tkinter interface with image preview and results display
- **Robust CNN Architecture**: 4-layer convolutional neural network with batch normalization and dropout
- **High Accuracy Model**: Enhanced training with data augmentation and learning rate scheduling
- **Educational Purpose**: Complete code with detailed comments for learning and research

## 📁 File Structure

```
Tomato_leaves_Disease_Detection_using_cnn/
├── app/                                    # Main application folder
│   └── gui.py                             # Tkinter GUI application
├── data/                                  # Training dataset
│   ├── train/                            # Training images organized by disease class
│   └── val/                              # Validation images organized by disease class
├── Testing_Set/                          # Sample test images
├── enhanced_4conv_model_with_save.py     # CNN model training script
├── tomato_leaf_final_model.h5           # Trained model file
├── README.md                            # Project documentation
└── requirements.txt                     # Python dependencies
```

## 🔧 Installation

### Prerequisites
- Python 3.7 or higher
- At least 4GB RAM (8GB recommended for training)
- CUDA-compatible GPU (optional, for faster training)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/tomato-leaf-disease-detection.git
   cd tomato-leaf-disease-detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv tomato_env
   source tomato_env/bin/activate  # On Windows: tomato_env\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the trained model:**
   - Ensure `tomato_leaf_final_model.h5` is in the project root directory
   - If not available, train the model using `enhanced_4conv_model_with_save.py`

## 🖥️ Usage

### Running the GUI Application

1. **Navigate to the app directory:**
   ```bash
   cd app
   ```

2. **Launch the application:**
   ```bash
   python gui.py
   ```

3. **Using the Application:**
   - Click "Upload Image" to select a tomato leaf image
   - Supported formats: JPG, JPEG, PNG, BMP, TIF, TIFF
   - View the prediction result below the image
   - Use "Reset" to clear the current image and result
   - Click "About" for project information

### Training a New Model

1. **Prepare your dataset:**
   - Organize images in `data/train/` and `data/val/` folders
   - Create subfolders for each disease class
   - Ensure balanced dataset with sufficient images per class

2. **Run the training script:**
   ```bash
   python enhanced_4conv_model_with_save.py
   ```

3. **Monitor training:**
   - Training runs for 20 epochs with automatic learning rate reduction
   - View accuracy and loss plots during training
   - Model automatically saves as `tomato_leaf_final_model.h5`

## 🧠 Model Architecture

### CNN Structure
- **Input Layer**: 128x128x3 RGB images
- **Convolutional Blocks**: 4 blocks with increasing filters (16→32→64→128)
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Regularization**: Batch Normalization and Dropout (0.3)
- **Optimization**: Adam optimizer with adaptive learning rate

### Training Configuration
- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 0.0002 (with ReduceLROnPlateau callback)
- **Data Augmentation**: Rotation (10°), Zoom (5%), Shift (5%), Horizontal flip
- **Loss Function**: Categorical crossentropy

## 📊 Disease Classes

The model can identify the following tomato leaf conditions:

1. **Bacterial Spot** - Caused by Xanthomonas bacteria
2. **Early Blight** - Fungal disease affecting older leaves
3. **Late Blight** - Serious fungal disease causing rapid plant death
4. **Leaf Mold** - Common in greenhouse conditions
5. **Septoria Leaf Spot** - Fungal spots with dark borders
6. **Spider Mites** - Pest damage causing stippling
7. **Target Spot** - Circular lesions with target-like appearance
8. **Yellow Leaf Curl Virus** - Viral disease causing leaf curling
9. **Mosaic Virus** - Viral disease causing mottled patterns
10. **Healthy** - Normal, disease-free leaves

## 📈 Model Performance

### Training Results
- **Training Accuracy**: ~92%
- **Validation Accuracy**: ~90%
- **Training Time**: Training time depends on hardware (~15–20 minutes on GPU, longer on CPU).
- **Model Size**: ~2.5MB (efficient for deployment)

### Evaluation Metrics
The model provides comprehensive evaluation including:
- Confusion Matrix visualization
- Classification Report with precision, recall, F1-score
- Per-class accuracy analysis
- Training/validation loss curves

## 🛠️ Technical Requirements

### Hardware Requirements
- **Minimum**: 4GB RAM, Intel i3 or equivalent
- **Recommended**: 8GB RAM, Intel i5 or equivalent, NVIDIA GPU
- **Storage**: 2GB free space for model and dependencies

### Software Dependencies
- TensorFlow 2.8+ (deep learning framework)
- NumPy (numerical computing)
- Pillow (image processing)
- Tkinter (GUI framework - usually included with Python)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (evaluation metrics)

## 🚀 Future Enhancements

### Planned Features
- **Confidence Scores**: Display prediction confidence percentages
- **Batch Processing**: Process multiple images simultaneously
- **CSV Export**: Export results for further analysis
- **Treatment Recommendations**: Suggest remedies for detected diseases
- **Mobile App**: React Native or Flutter mobile version
- **Web Interface**: Streamlit or Flask web application
- **Real-time Camera**: Live camera feed processing

### Technical Improvements
- **Model Optimization**: Quantization for faster inference
- **Transfer Learning**: Fine-tuning pre-trained models (ResNet, EfficientNet)
- **Data Pipeline**: Automated data collection and preprocessing
- **Cloud Deployment**: AWS/Google Cloud model serving

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for any changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset derived from PlantVillage (Tomato subset) with augmentation** for providing the tomato disease images
- **TensorFlow/Keras Team** for the excellent deep learning framework
- **Agricultural Research Community** for disease classification standards
- **Open Source Contributors** who make projects like this possible


## 📚 References & Resources

1. **Dataset Source**: PlantVillage - Tomato Disease Classification
2. **Deep Learning**: "Deep Learning for Plant Disease Detection" - Agricultural AI Research
3. **CNN Architecture**: TensorFlow Official Documentation
4. **Agricultural Applications**: "Machine Learning in Agriculture" - Research Papers
5. **Disease Information**: Agricultural Extension Services and Plant Pathology Resources

---

**Note**: This project is designed for educational and research purposes. For commercial agricultural applications, please consult with agricultural experts and consider additional validation with local crop conditions.
