import pandas as pd 
import numpy as np 
from Perceptron import *

print("==" * 50 ) 
print("Loading dataset...")




s = 'https://archive.ics.uci.edu/ml/'\
      'machine-learning-databases/iris/iris.data'
print('From URL:', s)
df = pd.read_csv(s,
                   header=None,
                  encoding='utf-8')
print(df.tail())

y = df.iloc[0:100, 4].values 
y = np.where(y == "Iris-setosa", 0 , 1 )

X = df.iloc[0:100 , [0,2]].values 

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='s', label='Versicolor')



#####################33
#####################33

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()



ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_,
          marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()