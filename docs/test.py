Python 3.4.0 (v3.4.0:04f714765c13, Mar 16 2014, 19:25:23) [MSC v.1600 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> mat = np.mat(([1,2,3],[4,5,6]))
>>> print(mat)
[[1 2 3]
 [4 5 6]]
>>> map = np.zeros((2,3))
>>> mat = np.zeros((2,3))
>>> print(mat)
[[0. 0. 0.]
 [0. 0. 0.]]
>>> mat = np.ones((2,3))
>>> print(mat)
[[1. 1. 1.]
 [1. 1. 1.]]
>>> 
>>> mat = np.random.rand(2,3)
>>> print(mat)
[[0.23867781 0.28925025 0.13540214]
 [0.05363748 0.53319775 0.7791956 ]]
>>> mat = np.eyes(3)
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    mat = np.eyes(3)
AttributeError: 'module' object has no attribute 'eyes'
>>> mat = np.eyes((3))
Traceback (most recent call last):
  File "<pyshell#12>", line 1, in <module>
    mat = np.eyes((3))
AttributeError: 'module' object has no attribute 'eyes'
>>> mat = np.eye(3)
>>> print(mat)
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
>>> 
