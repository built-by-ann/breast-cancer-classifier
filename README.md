# breast-cancer-classifier

A supervised machine learning system to classify breast tumors as **malignant** or **benign** using numerical cell nucleus features.

---

## Project Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate diagnosis can significantly improve patient outcomes. This project explores how classical and neural machine learning models can help in classifying tumors from biopsy data.

Using the **Breast Cancer Wisconsin Diagnostic Dataset**, I built and compared two classification pipelines:

- A Random Forest Classifier (RFC)
- A Sequential Neural Network (NN) using TensorFlow/Keras

Both models were trained and evaluated on 30 numerical features describing characteristics of cell nuclei present in digitized biopsy images.

---

## Goals

- Classify tumors as benign or malignant
- Compare performance of RFC vs. NN
- Evaluate precision, recall, and accuracy on real clinical data
- Build a reusable pipeline for future medical ML tasks  

---

## Technical Stack

| Layer         | Tech                        |
|---------------|-----------------------------|
| ML Models     | scikit-learn (RFC), Keras (NN) |
| Dataset       | sklearn’s breast cancer dataset |
| Preprocessing | NumPy, pandas, StandardScaler |
| Evaluation    | classification_report, accuracy_score |
| Visualization | matplotlib (optional) |

---

## Pipeline Progress

Dataset  
Loaded 569 samples from `sklearn.datasets`. Each sample includes 30 features and a binary label: 0 = malignant, 1 = benign.

Preprocessing  
- RFC: Used raw features (tree-based models don't need scaling)  
- NN: Standardized features using `StandardScaler` for better convergence

Random Forest Classifier  
- Tuned with `GridSearchCV` (5-fold cross-validation)  
- Best config: `n_estimators=200`, `max_depth=None`, `min_samples_split=2`  
- Accuracy: **96.5%** on test set  
- Strong precision/recall for both tumor classes

Neural Network (Keras)  
- Architecture: 16 → 8 → 1 (sigmoid output)  
- Loss: binary crossentropy, Optimizer: Adam  
- Trained for 50 epochs, batch size = 16  
- Accuracy: **98%** on test set  
- Precision: 0.98 (malignant), 0.99 (benign)

---

## Project Structure

```
breast-cancer-ai/
├── breast_cancer_classification.py   # Random Forest pipeline
├── breast_cancer_neural_net.py       # Neural Network pipeline
├── README.md                         # ← You are here
```

---

## Example Output

Random Forest  
```
Accuracy: 96.5%
Precision/Recall (Benign): ~0.97
Precision/Recall (Malignant): ~0.96
```

Neural Network  
```
Accuracy: 98.0%
Precision (Benign): 0.99
Precision (Malignant): 0.98
```

---

## To-Do / Roadmap

- Add learning curves and loss plots
- Integrate additional model types (e.g. SVM, XGBoost)
- Test on noisy/imbalanced variants of the dataset
- Deploy interactive UI for user-submitted test cases  

---

## References

- [Scikit-learn Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
- [TensorFlow Keras Sequential Model](https://www.tensorflow.org/guide/keras/sequential_model)  
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  

