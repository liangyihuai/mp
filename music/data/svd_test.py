import numpy as np;
from numpy import linalg as la;




a = np.matrix([[5, 2, 4, 4, 3], [3, 1, 2, 4, 2], [2, 0, 3, 1, 4], [2, 5, 4, 3, 5], [4, 4, 5, 4, 0]]);
U, sigma, VT = la.svd(a, full_matrices=False);
print("u");
print(U);
print("sigma: ");
print(sigma);
print ('vt');
print(VT);

S = np.diag(sigma)
# print(np.allclose(a, np.dot(U, np.dot(S, VT))))
print(np.dot(U, np.dot(S, VT)));

# print (np.dot(U[2].T, np.dot(S, VT)[1]));
# arr = np.zeros([34403, 419839]);
# print(arr);






