import torch
import sys
sys.path.append('/Users/lindseyschweitzer/Documents/GitHub/lfschweitzer.github.io/')

from posts.post5.logistic import GradientDescentOptimizer

class NewtonOptimizer(GradientDescentOptimizer):
    
    def __init__(self, model):
        self.model = model
    
    def hessian(self, X):
        
        s = self.model.score(X)
        sig_s = torch.sigmoid(s)

        d_diag = sig_s * (1 - sig_s)
        D = torch.diag(d_diag)
        
        H = X.T @ D @ X
        
        return H
    
    def step(self, X, y, alpha):
        
        H = self.hessian(X)
        grad = self.model.grad(X, y)

        self.model.w = self.model.w - (alpha * torch.linalg.inv(H))@ grad
        
