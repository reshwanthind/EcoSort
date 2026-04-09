# ♻️ EcoSort : Garbage Classification using Convolutional Neural Networks (CNN)

A deep learning-based image classification system that categorizes waste into six classes: cardboard, glass, metal, paper, plastic, and trash. The project demonstrates an end-to-end CNN pipeline with data augmentation, GPU memory optimization, early stopping, model checkpointing, and detailed evaluation using a confusion matrix and classification report.

---

## 🎯 Objective

- Develop an automated system for accurate garbage classification
- Reduce human effort and error in waste segregation
- Improve efficiency of recycling and waste management systems
- Apply CNNs to a practical environmental problem

---

## 🚀 Features

- 🧠 Custom CNN architecture for multi-class classification
- 🔄 Data augmentation for better generalization
- ⚡ GPU memory growth enabled for stable training
- 📉 Adaptive learning rate scheduling with `ReduceLROnPlateau`

---

## 📂 Dataset

Dataset source: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

### Structure

```text
dataset/
└── Garbage classification/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

- Images are organized class-wise
- Training/validation split is created automatically using `validation_split=0.2`

---

## 🏗️ Model Architecture

- Input: `224×224` RGB images
- 4 Convolutional blocks:
  - `Conv2D → MaxPooling2D → BatchNormalization`
- Fully connected layers:
  - `Dense(256) → Dropout(0.5) → Dense(128)`
- Output layer:
  - `Dense(6, activation='softmax')`

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## 🧪 Training Configuration

| Parameter | Value |
|---|---:|
| Image Size | 224 × 224 |
| Batch Size | 16 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Validation Split | 20% |

---

## 📈 Results

### Model Performance

- **Validation accuracy:** approximately **75%**
- **Weighted F1-score:** **0.75**
- **Macro F1-score:** **0.72**
- Best class performance was observed for **paper** with recall of **0.94**
- Hardest class to classify was **trash**, with lower precision and recall than the other categories

### Classification Report Summary

- cardboard: precision 0.90, recall 0.68, F1-score 0.77
- glass: precision 0.80, recall 0.78, F1-score 0.79
- metal: precision 0.73, recall 0.70, F1-score 0.71
- paper: precision 0.74, recall 0.94, F1-score 0.83
- plastic: precision 0.70, recall 0.66, F1-score 0.68
- trash: precision 0.52, recall 0.56, F1-score 0.54

### Generated Evaluation Outputs

- `results/training_metrics.png` → training and validation accuracy/loss curves
- `results/confusion_matrix.png` → confusion matrix heatmap
- `results/classification_report.txt` → precision / recall / F1 report
- `results/best_model.keras` → best checkpoint during training
- `results/garbage_model_high_acc.h5` → final saved model

---

## ▶️ Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/reshwanthind/EcoSort.git
cd EcoSort
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

- Go to the Kaggle dataset page: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Download and extract it
- Place the extracted folder inside the project as:

```text
dataset/Garbage classification/
```

> Note: `dataset/` is excluded from Git using `.gitignore`.

### 4. Run Training

```bash
python model.py
```

---

## 📊 Visualizations

The project automatically generates:

- Accuracy vs Epochs
- Loss vs Epochs
- Confusion Matrix
- Classification Report

All outputs are saved inside the `results/` directory.

---

## 📌 Future Improvements

- Transfer learning with models like ResNet, EfficientNet, or MobileNet
- Hyperparameter tuning to improve validation accuracy
- Deployment as a web app using Flask, FastAPI, or Streamlit
- Mobile or edge-device inference optimization

---

## 📝 License

This project is licensed under the MIT License.

