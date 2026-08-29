# Personalized Medical Recommendation System with Machine Learning

A Machine Learning-based medical recommendation system that analyzes user symptoms, predicts potential diseases, and provides personalized health-related recommendations through a Streamlit web application.

The system combines **Machine Learning, data preprocessing, disease classification, and recommendation logic** to create an easy-to-use healthcare assistance platform.

> **Disclaimer:** This project is for educational and demonstration purposes only. It is not intended to replace professional medical diagnosis, treatment, or advice.

---

## Features

- **Symptom-Based Disease Prediction**
  - Predicts potential diseases based on user-provided symptoms.

- **Multiple Machine Learning Models**
  - Support Vector Classifier (SVC)
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - Multinomial Naive Bayes

- **Model Evaluation**
  - Accuracy Score
  - Confusion Matrix
  - Comparison of multiple classification algorithms

- **Personalized Recommendations**
  - Top 5 medicine recommendations
  - Prescription-related information
  - Additional health recommendations

- **Workout Recommendations**
  - Provides workout and routine suggestions based on the predicted condition.

- **Streamlit Web Application**
  - Provides a user-friendly web interface for interacting with the system.

- **Privacy-Focused Design**
  - Designed with consideration for protecting user-provided health information.

---

## How It Works

The system follows a complete machine learning pipeline:

```text
                         User Symptoms
                               |
                               v
                      Data Preprocessing
                               |
                               v
                       Label Encoding
                               |
                               v
                       Train / Test Split
                               |
                               v
                +--------------------------+
                |   Machine Learning       |
                |        Models            |
                +--------------------------+
                |  SVC                     |
                |  Random Forest           |
                |  Gradient Boosting       |
                |  K-Nearest Neighbors     |
                |  Multinomial Naive Bayes |
                +--------------------------+
                               |
                               v
                       Disease Prediction
                               |
                               v
                 Personalized Recommendations
                               |
                  +------------+------------+
                  |            |            |
                  v            v            v
              Medicines   Prescription   Workout
```

## Machine Learning Workflow

### Load the Dataset

The system loads the training data using Pandas.

```python
import pandas as pd

dataset = pd.read_csv("Training.csv")
```

### Separate Features and Target

The symptom columns are used as input features, while `prognosis` is used as the target.

```python
X = dataset.drop("prognosis", axis=1)
y = dataset["prognosis"]
```

- `X` -> Input symptoms/features
- `y` -> Target disease/prognosis

### Encode Disease Labels

Disease names are converted into numerical values using `LabelEncoder`.

For example:

```text
Disease A -> 0
Disease B -> 1
Disease C -> 2
Disease D -> 3
```

This allows the machine learning algorithms to work with numerical class labels.

### Train-Test Split

The dataset is divided into:

```text
70% -> Training Data
30% -> Testing Data
```

using:

```python
train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=20
)
```

### Train Multiple Models

The system trains several classification algorithms using the same training data.

### Generate Predictions

After training, each model predicts diseases from the test dataset.

```python
predictions = model.predict(X_test)
```

### Evaluate the Models

The predictions are compared with the actual test labels using:

- Accuracy Score
- Confusion Matrix

---

## Machine Learning Models

The project compares five different classification algorithms.

| Model | Configuration |
|---|---|
| Support Vector Classifier | Linear Kernel |
| Random Forest | 100 Estimators |
| Gradient Boosting | 100 Estimators |
| K-Nearest Neighbors | 5 Neighbors |
| Multinomial Naive Bayes | Default Configuration |

### Model Configuration

```python
models = {
    "SVC": SVC(kernel="linear"),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),
    "KNeighbors": KNeighborsClassifier(
        n_neighbors=5
    ),
    "MultinomialNB": MultinomialNB()
}
```

---

## Model Evaluation

### Accuracy

Accuracy measures the percentage of correctly classified test samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

For example, if a model correctly predicts 90 out of 100 samples:

```text
Accuracy = 90 / 100 = 90%
```

### Confusion Matrix

The confusion matrix provides a detailed view of classification performance.

It shows:

- Correct predictions
- Incorrect predictions
- Classes that are confused with each other

The project generates a confusion matrix for every trained model.

---

The general flow is:

```text
User
 |
 v
Select / Enter Symptoms
 |
 v
Machine Learning Model
 |
 v
Predicted Disease
 |
 +---------------+
 |               |
 v               v
Medicine       Health
Recommendations Recommendations
 |
 +-- Medicines
 +-- Prescription Information
 +-- Workout Recommendations
```

---

## Recommendation System

After the system predicts a potential disease, it provides additional recommendation information, including:

- Top 5 medicine recommendations
- Prescription-related information
- Workout recommendations
- Additional health-related information

> These recommendations are intended for demonstration purposes and should not be treated as professional medical advice.

---

## Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Support Vector Machine
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Multinomial Naive Bayes

### Data Processing

- Pandas
- NumPy
- Label Encoding

### Development Environment

- Jupyter Notebook

---

- Streamlit

---

### Clone the Repository

```bash
git clone https://github.com/your-username/Personalized-Medical-Recommendation-System.git
```

### Navigate to the Project

```bash
cd Personalized-Medical-Recommendation-System
```

### Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Or install the main dependencies manually:

```bash
pip install pandas numpy scikit-learn streamlit
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

After starting the application, open the local URL displayed in the terminal.

---

## Run the Machine Learning Notebook

To experiment with the machine learning pipeline:

```bash
jupyter notebook
```

Then open:

```text
Medicine Recommendation System.ipynb
```

Run the notebook cells from top to bottom.

The notebook will:

1. Load the dataset
2. Explore the data
3. Prepare the features and target
4. Encode disease labels
5. Split the dataset
6. Train multiple ML models
7. Generate predictions
8. Calculate accuracy
9. Generate confusion matrices

---

## Example Output

The notebook evaluates every model and prints its performance.

```text
SVC Accuracy: 0.XX

SVC Confusion Matrix:
[...]

========================================

RandomForest Accuracy: 0.XX

RandomForest Confusion Matrix:
[...]

========================================

GradientBoosting Accuracy: 0.XX

GradientBoosting Confusion Matrix:
[...]

========================================

KNeighbors Accuracy: 0.XX

KNeighbors Confusion Matrix:
[...]

========================================

MultinomialNB Accuracy: 0.XX

MultinomialNB Confusion Matrix:
[...]
```

The model with the best evaluation performance can be selected for the disease classification task.

---

## Project Objectives

The main objectives of this project are to:

- Build a symptom-based disease classification system.
- Apply multiple machine learning algorithms.
- Compare different classification models.
- Integrate machine learning with a Streamlit web application.
- Provide personalized recommendation information.
- Provide workout-related recommendations.
- Demonstrate an end-to-end machine learning workflow.

---

## Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Advanced machine learning and deep learning models
- More comprehensive model evaluation
- Larger and more diverse datasets
- Improved privacy and security
- Production deployment
- Mobile-friendly interface
- Continuous model improvement
- More extensive validation and testing

---

## Medical Disclaimer

This project is developed for **educational, research, and demonstration purposes only**.

The disease predictions and recommendations generated by this system should **not be considered professional medical advice or a medical diagnosis**.

Do not use the system to make decisions about medication, treatment, exercise, or other medical care without consulting a qualified healthcare professional.

---

## Author

### Nisar Ahmad Zamani

**Machine Learning & AI Enthusiast**