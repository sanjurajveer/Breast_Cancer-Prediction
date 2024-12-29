#using decision tree first then using random forest to improve the accuracy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('cancer.csv')
dataset.columns
#benign is B and Malignant is M

# prompt: convert diagonosis column in 0 and 1
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
#removing unwanted columns
dataset=dataset.drop(columns=['id','Unnamed: 32'])


# Generate a correlation matrix
correlation_matrix = dataset.corr()
# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Find features highly correlated with the target
target_correlation = correlation_matrix['diagnosis'].abs().sort_values(ascending=False)
print("Features correlated with the target (diagnosis):\n", target_correlation)

# Remove features with high multicollinearity (absolute correlation > 0.9)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_correlation_pairs = [(column, row) for column in upper_triangle.columns for row in upper_triangle.index 
                          if abs(upper_triangle[column][row]) > 0.9]

print("\nHighly correlated feature pairs (correlation > 0.9):")
for pair in high_correlation_pairs:
    print(pair)

#Retain features that have a moderate to high correlation with the target variable (∣correlation∣>0.3)
#Drop those columns which are highly corr with each other |corr|>0.9
X = dataset.drop(columns=['symmetry_se', 'texture_se','fractal_dimension_mean','smoothness_se','fractal_dimension_se','concavity_se','compactness_se'])
X=dataset.drop(columns=['diagnosis','concavity_mean','perimeter_se','area_se','texture_mean','area_mean','perimeter_mean'])
X.columns

Y = dataset['diagnosis']
Y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling is not used when:
#
#Tree-based Algorithms (e.g., Decision Trees, Random Forest, Gradient Boosted Trees) are used as they 
#do not require scaling since they split data based on feature thresholds.
#If all features are already normalized or comparable in scale.

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

#importing required libararies
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Initialize the Decision Tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

# Train the classifier on the training data
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

'''
Confusion Matrix:
[[85  5]
 [ 1 52]]
Accuracy: 95.80%
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.94      0.97        90
           1       0.91      0.98      0.95        53

    accuracy                           0.96       143
   macro avg       0.95      0.96      0.96       143
weighted avg       0.96      0.96      0.96       143
'''
#to see the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(classifier, feature_names=dataset.columns[2:], class_names=['Benign', 'Malignant'], filled=True)
plt.show()


###Now using random forest to improve accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Initialize Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

'''
Confusion Matrix:
 [[87  3]
 [ 2 51]]
Accuracy: 96.50%
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.97      0.97        90
           1       0.94      0.96      0.95        53

    accuracy                           0.97       143
   macro avg       0.96      0.96      0.96       143
weighted avg       0.97      0.97      0.97       143

'''


