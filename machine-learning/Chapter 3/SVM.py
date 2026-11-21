from sklearn.svm import SVC
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from idmlm import *

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Use only petal length and petal width
y = iris.target

# Step 1: Split the data FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                     random_state=1, stratify=y)

# Check label distributions
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Step 2: Standardize the features
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

# Fixed plotting
plot_decision_regions(X_train_std,
                      y_train,  # Use training labels
                      classifier=svm)  # Remove invalid test_idx

plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()