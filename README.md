# 🌸 Iris MLP Classification

A machine learning project that classifies Iris flower species using a **Multi-Layer Perceptron (MLP) Neural Network** with hyperparameter tuning via GridSearchCV.

---

## 📌 Project Overview

This project uses the famous **Iris dataset** to build and evaluate a Neural Network classifier using Scikit-Learn.

The workflow includes:

- Data loading
- Exploratory Data Analysis (EDA)
- Train-test split
- Feature scaling
- MLP model training
- Hyperparameter tuning using GridSearchCV
- Model evaluation
- Model saving with Pickle

---

## 📊 Dataset

- Built-in dataset: `sklearn.datasets.load_iris()`
- Total samples: 150
- Classes:
  - Setosa
  - Versicolor
  - Virginica
- Features:
  - Sepal length
  - Sepal width
  - Petal length
  - Petal width

---

## 🧠 Model Used

### MLPClassifier (Multi-Layer Perceptron)

- Feedforward neural network
- Backpropagation training
- Hyperparameter tuning using GridSearchCV

---

## ⚙️ Project Workflow

1. Import required libraries
2. Load dataset
3. Convert to pandas DataFrame
4. Data inspection and analysis
5. Train-test split
6. Feature scaling using StandardScaler
7. Train MLP model
8. Perform GridSearchCV
9. Evaluate model performance
10. Save trained model using pickle

---

## 📈 Evaluation Metrics

- Accuracy Score
- Classification Report
- Confusion Matrix

Example evaluation:

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 💾 Model Saving

Save the trained model:

```python
import pickle

pickle.dump(model, open("iris_mlp_model.pkl", "wb"))
```

Load the model later:

```python
model = pickle.load(open("iris_mlp_model.pkl", "rb"))
```

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/iris-mlp-classification.git
```

2. Navigate to project folder

```bash
cd iris-mlp-classification
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Open Jupyter Notebook

```bash
jupyter notebook
```

5. Run `iris_mlp_classification.ipynb`

---

## 📁 Project Structure

```
iris-mlp-classification/
│
├── iris_mlp_classification.ipynb
├── iris_mlp_model.pkl
├── requirements.txt
└── README.md
```

---

## 🎯 Future Improvements

- Add model comparison (Logistic Regression, SVM)
- Add cross-validation plots
- Add training vs validation curve
- Add confusion matrix heatmap
- Deploy using Streamlit

---

## 👨‍💻 Author

Your Name  
Machine Learning Enthusiast 🚀
