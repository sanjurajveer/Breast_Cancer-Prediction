# Breast Cancer Prediction Using Machine Learning

## Overview
This project focuses on predicting whether a tumor is benign or malignant using a dataset of breast cancer cases. Various machine learning models, such as Decision Trees and Random Forests, are applied to classify tumors based on their features. The project includes data preprocessing, feature selection, and performance evaluation.

---

## Dataset
The dataset used in this project contains information about breast cancer cases with features such as:
- Mean radius, texture, perimeter, area, smoothness, etc.
- Diagnosis (B = Benign, M = Malignant).

The dataset was preprocessed to:
1. Encode the `diagnosis` column into numerical values (0 for Benign and 1 for Malignant).
2. Remove redundant features based on correlation analysis.
3. Split the data into training and testing sets.
4. Apply feature scaling for better model performance.

---

## Project Structure
- **`cancer.csv`**: Dataset file.
- **`main.py`**: Main script containing the code for loading data, preprocessing, training models, and evaluating performance.
- **`README.md`**: This documentation file.
- **`requirements.txt`**: List of Python dependencies for the project.

---

## Key Steps

### 1. Data Preprocessing
- Mapped the `diagnosis` column to binary values (0 and 1).
- Identified and dropped highly correlated features using a correlation threshold of 0.9.
- Split the dataset into training (75%) and testing (25%) subsets.
- Scaled the features using `StandardScaler` to ensure uniformity.

### 2. Model Training
#### Decision Tree
- Criterion: Entropy
- Achieved 95.8% accuracy.

#### Random Forest
- Number of Trees: 100 (default)
- Achieved higher accuracy (~96.5%).

### 3. Performance Evaluation
- Used Confusion Matrix and Accuracy Score to evaluate model performance.
- Compared the Decision Tree and Random Forest models.

---

## Usage

### Prerequisites
Ensure you have Python 3.x installed along with the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`



## Results
- Decision Tree: 95.8% accuracy
- Random Forest: ~96.5% accuracy

Random Forest performed better due to its ensemble nature, reducing overfitting and improving generalization.

---

## Future Improvements
- Explore other machine learning algorithms (e.g., SVM, Gradient Boosting).
- Perform hyperparameter tuning for models to optimize performance.
- Include additional visualizations for better interpretability.

---

## License
This project is licensed under the MIT License.

---

## Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

---

## Acknowledgments
- The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
- Thanks to the developers of `scikit-learn` and related libraries for making machine learning accessible.

---

## Contact
For questions or suggestions, please reach out via:
- **Email**: your-email@example.com
- **GitHub**: [sanjurajveer](https://github.com/sanjurajveer)

