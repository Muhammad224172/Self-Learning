from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from idmlm import *
from sigmoid import *

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
            errors = (y-output)
            self.w_  += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_  += self.eta * 2.0 * errors.mean()
            loss =  loss = (-y.dot(np.log(output))- ((1- y).dot(np.log(1 - output))) / X.shape[0])
            self.losses_.append(loss)
        return self 
    
    def net_input(self , X): 

        return np.dot(X , self.w_) + self.b_
    
    def activation(self ,X): 
        return 1./(1+np.exp(-np.clip(z, -250, 250)))
    
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

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))


############

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,

                       y=y_combined,


                       classifier=ppn,
                       test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



#################################################################################
#################################################################################
################################################################################
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

#First Logistic Regression Model 
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta=0.3,
                              n_iter=1000,
                              random_state=1)
lrgd.fit(X_train_01_subset,
           y_train_01_subset)
plot_decision_regions(X=X_train_01_subset,
                        y=y_train_01_subset,
                        classifier=lrgd)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()