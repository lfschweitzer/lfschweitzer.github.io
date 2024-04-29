import torch
import sys
sys.path.append('/Users/lindseyschweitzer/Documents/GitHub/lfschweitzer.github.io/')

from posts.post5.logistic import GradientDescentOptimizer

class NewtonOptimizer(GradientDescentOptimizer):
    
    def __init__(self, model):
        self.model = model
    
    def hessian(self, X):
        '''
        Compute the Hessian matrix, which is the matrix of second derivatives of L.
        
        AGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s
        RETURNS:
            H, torch.Tensor: the Hessian matrix
        '''
        
        s = self.model.score(X)
        sig_s = torch.sigmoid(s)

        d_diag = sig_s * (1 - sig_s)
        D = torch.diag(d_diag)
        
        H = X.T @ D @ X
        
        return H
    
    def step(self, X, y, alpha):
        """
        Compute one step of the update using the feature matrix X, target vector y,
        and learning rate alpha.
        
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s
            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
            alpha: float: learning rate of model
        RETURNS:
            N/A
        """
        
        H = self.hessian(X)
        grad = self.model.grad(X, y)

        self.model.w = self.model.w - (alpha * torch.linalg.inv(H))@ grad
        
