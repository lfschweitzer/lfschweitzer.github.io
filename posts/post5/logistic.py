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
    
    def sigmoid(self, s):
        return 1/(1 + torch.exp(-s))
    
    
    def loss(self, X, y):
        """
        Should compute the empirical risk using the logistic loss function
        """
        s = self.score(X)
        
        # print(f"{s=}")
        
        sig_s = self.sigmoid(s)
        y = y[:, None]
        
        little_loss = -y * torch.log(sig_s) - (1-y) * torch.log(1-sig_s)
        
        # print(f"{little_loss=}")
        loss = torch.mean(little_loss)
        
        return loss

    def grad(self, X, y):
        """
        Should compute the gradient of the empirical risk
        """
        s = self.score(X)
        sig_s = self.sigmoid(s)
        
        little_grad = (sig_s - y)[:, None] * X
        
        grad = torch.mean(little_grad, dim=0)
        
        return grad 

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, alpha, beta):
        """
        Compute one step of the update using the feature matrix X 
        and target vector y. 
        """
        loss = self.model.loss(X, y)
        grad = self.model.grad(X, y)
        
        cur_w = self.model.w
        
        # if it is the first update
        if self.model.prev_w == None:
            self.model.w -= alpha*grad
        else:
            self.model.w += -1*alpha*grad + beta*(cur_w - self.model.prev_w)
        
        # save value of previous w
        self.model.prev_w = cur_w
        
        return loss