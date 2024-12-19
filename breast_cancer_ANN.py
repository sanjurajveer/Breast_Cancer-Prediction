# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('breast-cancer_csv.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X = X.astype(str)
y = y.reshape((len(y), 1))
#encoding cataegorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X_0 = LabelEncoder()
X[:, 0] = LabelEncoder_X_0.fit_transform(X[:, 0])
LabelEncoder_X_1 = LabelEncoder()
X[:, 1] = LabelEncoder_X_1.fit_transform(X[:, 1])
LabelEncoder_X_2 = LabelEncoder()
X[:, 2] = LabelEncoder_X_2.fit_transform(X[:, 2])
LabelEncoder_X_3 = LabelEncoder()
X[:, 3] = LabelEncoder_X_3.fit_transform(X[:, 3])
LabelEncoder_X_4 = LabelEncoder()
X[:, 4] = LabelEncoder_X_4.fit_transform(X[:, 4])
LabelEncoder_X_6 = LabelEncoder()
X[:, 6] = LabelEncoder_X_6.fit_transform(X[:, 6])
LabelEncoder_X_7 = LabelEncoder()
X[:, 7] = LabelEncoder_X_7.fit_transform(X[:, 7])
LabelEncoder_X_8 = LabelEncoder()
X[:, 8] = LabelEncoder_X_8.fit_transform(X[:, 8])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 0:] 
ct = ColumnTransformer([('encoder', OneHotEncoder(), [7])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:, 7:] 
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
# importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu', input_dim = 17))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu'))

# Adding the fifth hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu'))

# Adding the sixth hidden layer
classifier.add(Dense(output_dim = 9, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, verbose = 2)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
