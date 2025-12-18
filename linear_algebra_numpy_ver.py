"""
In this file I will be recreating the funcitons that we created to do linear 
algebra, but instead of using base python objects like lists I will be using
numpy. This will increase performance and make the functions operate smoother
when used in machine learning applications. 

Insert a quick word on Numpy here:
"""

import numpy as np

foo = np.array([[1,2],[3,4]])
bar = np.array([[1,2],[3,4]])

print(foo.ndim)
print(foo.shape)
print(foo @ bar)