# Breast Cancer Classification

This project evaluates the performance of five classification algorithms on the breast cancer dataset from the `sklearn` library. The goal is to predict whether a tumor is malignant or benign based on features extracted from medical imaging.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)
10. [Contact](#contact)

---

## Project Overview

The objective of this project is to evaluate classification techniques in supervised learning by applying them to the breast cancer dataset. The key steps include:

1. **Loading and Preprocessing**:
   - Load the dataset.
   - Handle missing values (if any).
   - Perform feature scaling.

2. **Classification Algorithm Implementation**:
   - Implement and train five classification models:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - Support Vector Machine (SVM)
     - k-Nearest Neighbors (k-NN)

3. **Model Evaluation and Comparison**:
   - Evaluate models using accuracy, precision, recall, and F1 score.
   - Compare the results to identify the best and worst-performing models.

---

## Dataset

The breast cancer dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. The features describe characteristics of the cell nuclei, such as:

- **Radius**: Mean of distances from the center to points on the perimeter.
- **Texture**: Standard deviation of gray-scale values.
- **Perimeter**: Perimeter of the cell nucleus.
- **Area**: Area of the cell nucleus.
- **Smoothness**: Local variation in radius lengths.
- **Compactness**: Perimeter² / area - 1.0.
- **Concavity**: Severity of concave portions of the contour.
- **Symmetry**: Symmetry of the cell nucleus.
- **Fractal Dimension**: "Coastline approximation" - 1.

The target variable is binary:
- **0**: Malignant (cancerous)
- **1**: Benign (non-cancerous)

### Dataset Characteristics
- **Number of instances**: 569
- **Number of features**: 30
- **Target variable**: Binary (0 or 1)
- **Missing values**: None

---

## Methodology

### Preprocessing

#### Feature Scaling:
- Features are standardized using **StandardScaler** to ensure all features have a mean of 0 and a standard deviation of 1.
- This is necessary for algorithms like **Logistic Regression** and **SVM**, which are sensitive to the scale of input features.

#### Train-Test Split:
- The dataset is split into **training (80%)** and **testing (20%)** sets to evaluate model performance on unseen data.

### Classification Algorithms

1. **Logistic Regression**:
   - Models the probability of the target variable using a logistic function.
   - Suitable for binary classification problems.

2. **Decision Tree Classifier**:
   - Splits data into subsets based on feature thresholds.
   - Captures non-linear relationships but is prone to overfitting.

3. **Random Forest Classifier**:
   - Ensemble of decision trees to reduce overfitting.
   - Handles non-linear relationships and feature interactions effectively.

4. **Support Vector Machine (SVM)**:
   - Finds the optimal hyperplane to separate classes.
   - Suitable for high-dimensional datasets.

5. **k-Nearest Neighbors (k-NN)**:
   - Classifies data points based on the majority class of their k-nearest neighbors.
   - Simple and effective for small datasets.

### Evaluation Metrics

- **Accuracy**: Measures the proportion of correctly classified instances.
- **Precision**: Measures the proportion of true positives among predicted positives.
- **Recall**: Measures the proportion of true positives among actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

## Results

### Model Comparison

| Model                   | Accuracy | Precision | Recall  | F1 Score |
|-------------------------|----------|-----------|---------|----------|
| Logistic Regression      | 0.9825   | 0.9859    | 0.9859  | 0.9859   |
| Decision Tree            | 0.9474   | 0.9474    | 0.9577  | 0.9525   |
| Random Forest            | 0.9649   | 0.9583    | 0.9859  | 0.9718   |
| Support Vector Machine   | 0.9825   | 0.9859    | 0.9859  | 0.9859   |
| k-Nearest Neighbors      | 0.9737   | 0.9718    | 0.9859  | 0.9787   |

### Best Model
- **Logistic Regression** and **Support Vector Machine** achieved the highest accuracy (0.9825).

### Worst Model
- **Decision Tree** had the lowest accuracy (0.9474).

## Conclusion
- **Logistic Regression** and **Support Vector Machine** performed the best due to their ability to model complex decision boundaries.
- **Decision Tree** performed the worst, likely due to overfitting on the training data.
- Ensemble methods like **Random Forest** and **k-NN** provided a good balance between accuracy and generalization.


## Acknowledgments

- **Dataset**: Breast Cancer Dataset from sklearn.
- **Libraries**: numpy, pandas, scikit-learn, matplotlib, and jupyter.
- **Inspiration**: This project was inspired by the need to evaluate and compare classification techniques for real-world datasets.

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Name**: Musabbir Km
- **Email**: musabbirmushu@gmail.com
- **GitHub**: [your-username](https://github.com/musabbirkm)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/MusabbirKm)


## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Steps to Set Up the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/musabbirkm/BreastCancerClassification.git
   cd BreastCancerClassification
   pip install -r requirements.txt
   
