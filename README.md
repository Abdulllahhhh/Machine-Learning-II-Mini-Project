# Machine Learning II Mini Project

## Description

This project was developed to explore and compare different machine learning approaches using Python.

Implemented algorithms:
- SVM
- Q-learning
- K-means
- Neural Networks

---

## Results

| Algorithm | Result |
|---|---|
| SVM Accuracy | 95.5% |
| Cross Validation Score | 98.0% |
| Neural Network Accuracy | 93.3% |
| K-means Clusters | 3 Clusters |
| Q-learning Reward | Improved after 200 iterations |

---

## SVM Example Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
```

---

## Tools

- Python
- NumPy
- scikit-learn

---

## Full Report

[View Project Report PDF](./Machine%20Learning%20II%20Mini%20Project.pdf)
