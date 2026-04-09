# ♻️ EcoSort : Garbage Classification using Convolutional Neural Networks (CNN)

A deep learning-based image classification system that categorizes waste into six classes: cardboard, glass, metal, paper, plastic, and trash. The project focuses on building an efficient CNN pipeline with strong generalization using data augmentation and GPU optimization.

---

## 🎯 Objective

- Develop an automated system for accurate garbage classification
- Reduce human effort and error in waste segregation
- Improve efficiency of recycling and waste management systems
- Leverage deep learning (CNNs) for real-world environmental impact

---

## 🚀 Features

- 🧠 Custom CNN architecture for multi-class classification
- 🔄 Data augmentation for better generalization
- 📉 Adaptive learning rate scheduling (ReduceLROnPlateau)
- 📊 Training visualization (accuracy & loss curves)
- 💾 Model saving for inference and deployment

---

## 📂 Dataset

Dataset source: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

### Structure

```
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
- Automatically split into 80% training and 20% validation

---

## 🏗️ Model Architecture

- Input: 224×224 RGB images
- Convolutional Blocks:
  - Conv2D → MaxPooling → BatchNormalization
- Fully Connected Layers:
  - Dense(256) → Dropout(0.5) → Dense(128)
- Output Layer:
  - Softmax (6 classes)

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🧪 Training Configuration

| Parameter  | Value                    |
| ---------- | ------------------------ |
| Image Size | 224 × 224                |
| Batch Size | 16                       |
| Epochs     | 50                       |
| Optimizer  | Adam                     |
| Loss       | Categorical Crossentropy |

---

## 📈 Results

- Achieved **~85% validation accuracy** on the garbage classification dataset
- Stable convergence with data augmentation
- Reduced overfitting using dropout and batch normalization
- Improved validation performance via dynamic learning rate scheduling

### Output Artifacts

- `results/garbage_model_high_acc.h5` → Trained model
- `results/training_metrics.png` → Accuracy & loss curves

---

## ▶️ Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/reshwanthind/garbage-classification-cnn.git
cd garbage-classification-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

- Go to the dataset link: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Download and extract it
- Place it inside the project as:

```
dataset/Garbage classification/
```

⚠️ Note: The dataset is ignored via `.gitignore`

### 4. Run Training

```bash
python model.py
```

---

## 📊 Visualization

Training generates:

- Accuracy vs Epochs
- Loss vs Epochs

Graphs are saved automatically in the `results/` directory.

---


## 📝 License

This project is licensed under the MIT License.