# COVID-19 Detection Using Chest X-ray Images 🦠🩻

This deep learning project focuses on detecting COVID-19 from chest X-ray images using a **Convolutional Neural Network (CNN)**. The goal is to assist in fast, automated, and non-invasive diagnosis of COVID-19 using medical imaging.

## 📌 Project Overview

- **Objective:** Classify chest X-ray images into two categories — COVID-19 Positive and Normal.
- **Dataset:** Publicly available dataset containing X-ray images of COVID-19 patients and healthy individuals.
- **Model:** Custom-built CNN using TensorFlow/Keras.
- **Evaluation Metrics:** Accuracy, Precision, Recall, Confusion Matrix

## 🧠 Technologies Used

- Python 🐍
- TensorFlow & Keras
- OpenCV
- NumPy & Matplotlib
- Scikit-learn

## 🗂️ Project Structure

```
covid19-xray-detection/
│
├── dataset/
│   ├── train/
│   │   ├── covid/
│   │   └── normal/
│   ├── test/
│       ├── covid/
│       └── normal/
│
├── covid_xray_classifier.ipynb     # Jupyter Notebook with full training and evaluation pipeline
├── model/                          # Saved model after training
│   └── covid_cnn_model.h5
├── images/                         # Example predictions and confusion matrix
│   └── result.png
├── requirements.txt                # All dependencies
└── README.md                       # Project documentation
```

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aarya-04/covid19-xray-detection.git
   cd covid19-xray-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook covid_xray_classifier.ipynb
   ```

4. **Follow notebook cells to:**
   - Load and preprocess image data
   - Train the CNN model
   - Evaluate accuracy and plot confusion matrix
   - Make predictions on new images

## 📊 Model Summary

- **Architecture:**
  - 3 Convolutional layers
  - Max Pooling layers
  - Fully connected dense layers
  - Dropout for regularization
- **Activation:** ReLU, Softmax
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy

## 🖼️ Sample Output

- Model Accuracy: ~95%
- Confusion Matrix:
  ![Confusion Matrix](images/result.png)

## 📁 Dataset Source

- COVID-19 Radiography Database: [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

> Note: Make sure to download and organize the dataset as per the folder structure shown above.

## 📌 Future Improvements

- Include multi-class classification (e.g., COVID, Pneumonia, Normal)
- Integrate Grad-CAM for visual model interpretability
- Build a web interface using Streamlit

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👨‍💻 Author

**Aarya Kulkarni**  
[LinkedIn](https://www.linkedin.com/in/aaryakulkarni03) • [GitHub](https://github.com/Aarya-04)

---

> Deep learning meets medical imaging — enabling fast COVID-19 diagnosis with just an X-ray.
