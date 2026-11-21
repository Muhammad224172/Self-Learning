import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

s = 'https://archive.ics.uci.edu/ml/'\
      'machine-learning-databases/iris/iris.data'
print('From URL:', s)
df = pd.read_csv(s,
                   header=None,
                  encoding='utf-8')
print(df.tail())

y = df.iloc[0:100, 4].values 
y = np.where(y == "Irish-setosa", 0 , 1 )

X = df.iloc[0:100 , [0,2]].values 

plt.scatter(X[:50 , 0 ], X[:50, 1 ] ,
            color = "red" , marker = "o" , label = "Setosa") 

plt.scatter(X[50:100, 0 ], X[50:100, 1 ] ,
            color = "blue" , marker = "o" , label = "Versicolor") 
plt.xlabel("Sepal Length[cm]")

plt.ylabel("Petal length [cm]")

plt.legend(loc = "upper left")

plt.show()



