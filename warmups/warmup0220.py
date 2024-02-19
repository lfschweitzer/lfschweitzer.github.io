import numpy as np

def linear_classify(x, w, b):
    
    dotProduct = np.dot(x, w)
    
    if dotProduct < b:
        return 0
    else:
        return 1


x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
b = 5
print(linear_classify(x,y,b))

    