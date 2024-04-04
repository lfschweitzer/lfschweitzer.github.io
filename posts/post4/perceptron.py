import torch

class LinearModel:

    def __init__(self):
        self.w = None 

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

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. In the perceptron algorithm, the target vector y is assumed to have labels in {-1, 1}. A point i is classified correctly if its score s_i has the same sign as y_i. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). In the perceptron algorithm, the possible labels for y are assumed to be {-1, 1}
        
        RETURNS: 
            loss: float: the loss value of the model
        """
        
        y_hat = self.predict(X)
        
        # convert y_hat from {0, 1} to {-1, 1}
        y_hat = 2*y_hat - 1
        
        misc = 1.0*(y_hat*y <= 0)
                        
        return misc.mean()

    def grad(self, X, y):
        """
        Should compute the gradient of the empirical risk
        
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

        RETURNS:
            grad: float: the gradient of the empirical risk of the model
        """
        s = self.score(X)
        
        # choose random learning rate
        learning_rate = 0.01

        # if misclassified, calculate update
        misclass = s*y <= 0
        update_val_row = X*y[:,None]
        
        update_val = update_val_row * misclass[:,None]
        
        r = learning_rate * torch.mean(update_val, 0)
        
        return r
       
class PerceptronOptimizer:

    def __init__(self, model):
        
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the update using the feature matrix X 
        and target vector y. 
        
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
            alpha: float: learning rate of model
            beta: float: momentum of model
        RETURNS:
            loss: float: the empirical risk
        """
        loss = self.model.loss(X, y)
        
        grad = self.model.grad(X, y)
        self.model.w += grad
        
        return loss