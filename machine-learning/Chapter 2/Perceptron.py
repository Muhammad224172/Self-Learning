import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors  import ListedColormap

class Perceptron: 
    
    def __init__(self , eta = 0.01 , n_iter = 50 , random_state = 1):
        self.eta = eta 
        self.n_iter = n_iter 
        self.random_state = random_state 

    def fit(self , X , y ): 

        rgen = np.random.RandomState(self.random_state)

        self.w_ = rgen.normal(loc = 0.0 , scale = 0.01 , size = X.shape[1])

        self.b_ = np.float_(0.)
        self.errors_ = []

        for _  in range(self.n_iter): 
            errors = 0 
            for xi , target in zip(X, y): 
                update = self.eta * (target - self.predict(xi))
                self.w_ += update *xi 
                self.b_ += update 
                errors += int(update != 0.0 )
            self.errors_.append(errors)
        return self 
    
    def net_input(self , X): 

        return np.dot(X , self.w_) + self.b_
    
    def predict(self , X): 

        return np.where(self.net_input(X) >= 0.0 , 1 , 0 )
    
class AdalineGD: 
    
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
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self 
    
    def net_input(self , X): 

        return np.dot(X , self.w_) + self.b_
    
    def activation(self ,X): 

        return X
    
    def predict(self , X): 

        return np.where(self.net_input(X) >= 0.5 , 1 , 0 )


class AdalineSGD: 
    
    def __init__(self ,shuffle = True , eta = 0.01 , n_iter = 50 , random_state = 1):
        self.eta = eta 
        self.n_iter = n_iter 
        self.random_state = random_state 
        self.w_initialized = False 
        self.shuffle = shuffle 
        

    def fit(self , X , y ): 

        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _  in range(self.n_iter): 

            if self.shuffle: 
                X,y = self._shuffle(X,y)
            losses = []
            for xi , target in zip(X,y): 
                losses.append(self._update_weights(xi , target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self 
    
    def partial_fit(self, X, y):

        if not self.w_initialized:
                    self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
                self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss
   
    
    def net_input(self , X): 

        return np.dot(X , self.w_) + self.b_
    
    def activation(self ,X): 

        return X
    
    def predict(self , X): 

        return np.where(self.net_input(X) >= 0.5 , 1 , 0 )


def plot_decision_regions(X, y , classifier , resolution = 0.02): 
    markers = ('o' , 's' , '^' , 'v' , '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min , x1_max = X[:,0].min()-1 , X[:,0].max()+1 
    x2_min , x2_max = X[:,1].min()-1 , X[:,1].max()+1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

        