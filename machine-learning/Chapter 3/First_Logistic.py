from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from idmlm import *
from sigmoid import *
from sklearn.linear_model import LogisticRegression

class LogisticRegressionGD: 
    
    def __init__(self , eta = 0.01 , n_iter = 50 , random_state = 1):
        self.eta = eta 
        self.n_iter = n_iter 
        self.random_state = random_state 
        

    def fit(self , X , y ): 

        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc = 0.0 , scale = 0.01 , size = X.shape[1])

        self.b_ = np.float_(0.)
        self.losses_ = []

        for _  in range(self.n_iter): 
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (output - y )
            self.w_  -= self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_  -= self.eta * 2.0 * errors.mean()
            loss  = (-y.dot(np.log(output))- ((1- y).dot(np.log(1 - output))) / X.shape[0])
            self.losses_.append(loss)
        return self 
    
    def net_input(self , X): 

        return np.dot(X , self.w_) + self.b_
    
    def activation(self ,X): 
        return 1./(1+np.exp(-np.clip(X, -250, 250)))
    
    def predict(self , X): 

        return np.where(self.net_input(X) >= 0.5 , 1 , 0 )
    
    
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print("Class labels: " , np.unique(y))


X_train, X_test, y_train, y_test = train_test_split(X , y , test_size= 0.3 , random_state=1 , stratify=y)

print('Labels counts in y:', np.bincount(y))
      
print('Labels counts in y_train:', np.bincount(y_train))
            
print('Labels counts in y_test:', np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# #First Logistic Regression Model 
# X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
# y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
# lrgd = LogisticRegressionGD(eta=0.3,
#                               n_iter=1000,
#                               random_state=1)
# lrgd.fit(X_train_01_subset,
#            y_train_01_subset)
# plot_decision_regions(X=X_train_01_subset,
#                         y=y_train_01_subset,
#                         classifier=lrgd)
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

#####################################################
#####################################################
#scikit Learn Implementation \ >>> from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load iris dataset
iris = datasets.load_iris()
X = iris.data  # All 4 features
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

# Step 3: Fit the model AFTER creating the training data
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

# Option A: Plot using only 2 features for visualization
X_train_2d = X_train_std[:, [2, 3]]  # Use petal length and width for plotting
lr_2d = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr_2d.fit(X_train_2d, y_train)

# Assuming you have a plot_decision_regions function:
plot_decision_regions(X_train_2d, y_train, classifier=lr_2d)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Option B: If you want to use all features, you can't easily visualize decision regions
# But you can evaluate the model:
train_accuracy = lr.score(X_train_std, y_train)
test_accuracy = lr.score(X_test_std, y_test)

print(f'Training accuracy: {train_accuracy:.3f}')
print(f'Test accuracy: {test_accuracy:.3f}')


#we can predict the probabilities of the first three examples in the test 
#dataset as follows
print(lr.predict_proba(X_test_std[:3, :]))

#we can get the predicted class labels by identifying the largest column in each row, for ex
#ample, using NumPy’s argmax function
print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))
print(lr.predict(X_test_std[:3, :]))



weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c,multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
weights = np.array(weights) 
plt.plot(params, weights[:, 0],label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--',label='Petal width') 
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()