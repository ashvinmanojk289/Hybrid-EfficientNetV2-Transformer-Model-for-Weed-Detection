# 🌿 Weed Detection & Deep Learning Comparison

![Agricultural Field](https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcRwjbHPPxo2vTBhEyWTJxGliEhoO6BHft1oMKDTCrB-JlomTKaxO5BcoBtkom6dHtamMI-_G8pUPJOXl-yvTu9TZIlangO_pFQdaymeS8sWE7LSe3Q)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)  
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 Project Overview
This project provides an **end-to-end pipeline** for tackling the challenge of **weed detection in agriculture** using computer vision.  
It trains, evaluates, and compares four powerful deep learning models to determine the most effective architecture for distinguishing between crops and weeds.  

Implemented using **TensorFlow** and **Keras**, this repository includes:
- A standard CNN  
- A Vision Transformer (ViT)  
- A Hybrid CNN-Transformer  
- An EfficientNetV2-based Transformer  

---

## 🎯 Key Features
- 🧠 **Four Model Architectures**: CNN, Vision Transformer, Hybrid CNN-Transformer, EfficientNetV2-Transformer  
- ⚙ **End-to-End Pipeline**: Data preparation → preprocessing → training → evaluation → visualization  
- 📊 **Robust Evaluation**: 5 training runs per model for statistical reliability  
- 📈 **Detailed Analysis**: Accuracy, loss, precision, recall, F1-score, and confusion matrices  
- 📦 **Modular & Reproducible**: Clean code structure with separate scripts  
- ⚡ **Powered by TensorFlow/Keras**: High performance and scalability  

---

## 🧩 System Architecture & Pipeline
```text
Download Raw Dataset 
   ↓
Prepare & Split Data 
   ↓
Load & Preprocess Batches 
   ↓
Define 4 Model Architectures 
   ↓
Train Each Model (5 Runs) 
   ↓
Evaluate on Test Set 
   ↓
Generate Reports, Plots & Summary
```

## 🏗️ Tech Stack

| Task | Model |
|------|-------|
| Language | Python |
| Web Framework | Streamlit |
| ML Framework | TensorFlow, Keras |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Utilities | Scikit-learn |


## ⚙️ Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Weed-Detection-Deep-Learning-Comparison.git
cd Weed-Detection-Deep-Learning-Comparison
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the Dataset
  - Obtain the weed detection dataset used in the project.
  - Unzip the file and place it in a convenient location.

5. Configure the Data Path
  - Open the `prepare_data.py` file.
  - Crucially, update the `ORIGINAL_DATASET_DIR` variable to point to the absolute path of the directory where you just unzipped the dataset.

6. **Run the pipeline**
```bash
# Step 1: Prepare data
python prepare_data.py

# Step 2: Train models (may take a long time)
python train.py

# Step 3: Evaluate models & generate results
python evaluate.py
```

## 🧪 Evaluation

The effectiveness of each model is rigorously evaluated to ensure a fair and comprehensive comparison.
- Quantitative Metrics: Performance is measured using standard classification metrics: Accuracy, Precision, Recall, and F1-Score (macro-averaged).
- Multi-Run Analysis: To account for randomness in model initialization and training, each model is trained 5 times. The final performance is reported as the mean and standard deviation of these runs.
- Qualitative Analysis: Confusion matrices are generated for each run to provide a visual understanding of how each model performs on a class-by-class basis.

## 🖼️ Application Screenshots
The training and evaluation scripts automatically generate and save several key visualizations in the training_results directory, including:
1. Training & Validation History: Line plots showing the accuracy and loss for each epoch.
2. Confusion Matrix: Heatmaps illustrating the model's predictions versus the actual labels on the test data.

## 🤝 Contributing
We welcome contributions!  
If you have suggestions or improvements, feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## 📝 Future Work

- Hyperparameter Tuning: Implement automated tuning (e.g., KerasTuner, Optuna) to find the optimal set of hyperparameters for each model.
- Object Detection: Extend the project from classification to object detection to draw bounding boxes around weeds.
- Deployment: Containerize the best-performing model with Docker and deploy it as a web service or API.
- Data Augmentation: Experiment with more advanced data augmentation techniques to improve model generalization.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgements

- The TensorFlow and Keras teams for their exceptional deep learning frameworks.
- The creators of the agricultural image dataset used for this project.
- The open-source community for providing the tools that made this work possible.
