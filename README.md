# Diabetes Classification and Prediction

This project aims to classify and predict diabetes in patients using a Support Vector Machine (SVM) algorithm. SVM is a supervised machine learning model that can be used for both classification and regression challenges. It works by finding the hyperplane that best divides a dataset into classes.

## How the Code Works

1. **Data Preprocessing**: The dataset is first preprocessed to handle missing values, normalize features, and split into training and testing sets.
2. **Model Training**: An SVM model is trained on the training dataset. The SVM algorithm tries to find the optimal hyperplane that separates the data points of different classes (diabetic and non-diabetic) with the maximum margin.
3. **Prediction**: The trained SVM model is then used to predict the class labels of the test dataset.
4. **Evaluation**: The performance of the model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

The SVM algorithm is particularly effective for high-dimensional spaces and is robust against overfitting, especially in cases where the number of dimensions exceeds the number of samples.

## Example Code Snippet

Here is a simplified example of how an SVM might be used in Python with the `scikit-learn` library:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
```

# Load dataset
```python
data = pd.read_csv('diabetes.csv')
``` 
# Preprocess data

```python
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
```

## Dependencies

The following libraries are required to run the notebook:

- numpy
- pandas
- scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn
```

## Data Collection

The dataset used in this project is the PIMA Diabetes Dataset. It contains several medical predictor variables and one target variable indicating the presence of diabetes.

## Data Preprocessing

The data preprocessing steps include:

- Standardizing the features using `StandardScaler`
- Splitting the dataset into training and testing sets using `train_test_split`

## Model Training

The Support Vector Machine (SVM) algorithm is used for training the model. The following steps are performed:

1. Importing the SVM classifier from scikit-learn
2. Training the model on the training data
3. Making predictions on the test data

## Model Evaluation

The accuracy of the model is evaluated using the `accuracy_score` metric.

## Usage

To run the notebook, open `Diabetes_Classification.ipynb` in Jupyter Notebook or Google Colab and execute the cells sequentially.


## Acknowledgements

- The PIMA Diabetes Dataset is provided by the National Institute of Diabetes and Digestive and Kidney Diseases.

