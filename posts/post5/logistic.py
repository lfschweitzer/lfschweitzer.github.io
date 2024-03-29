import torch
import numpy

class LinearModel:

    def __init__(self):
        self.w = None 
        self.prev_w = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
            
        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s = self.score(X)
        y_hat = 1.0*(s >= 0)
        
        return y_hat

class LogisticRegression(LinearModel):
    
    def sigmoid(s):
        return 1/(1+numpy.exp(-s))
    
    
    def loss(self, X, y):
        """
        Should compute the empirical risk using the logistic loss function
        """
        s = self.score(X)[:, None]
        sig_s = self.sigmoid(s)
        
        little_loss = -y[:, None]*numpy.log(sig_s) - (1-y)[:, None]* numpy.log(1-sig_s)
        
        return torch.mean(little_loss)

    def grad(self, X, y):
        """
        Should compute the gradient of the empirical risk
        """
        s = self.score(X)[:, None]
        sig_s = self.sigmoid(s)
        
        little_loss = (sig_s - y[:, None])*X
        
        return torch.mean(little_loss)

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the update using the feature matrix X 
        and target vector y. 
        """
        loss = self.model.loss(X, y)
        
        alpha = 0.5
        beta = 0.9
        
        grad = self.model.grad(X, y)
        
        # if it is the first update
        if self.prev_w == None:
            self.w += -alpha*grad

        else:
            cur_w = self.w
            self.w += -alpha*grad + beta*(cur_w - self.prev_w)
        
        # save value of previous w
        self.prev_w = cur_w
        
        return loss