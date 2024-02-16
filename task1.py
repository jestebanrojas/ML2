#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
## 1. Simulate any random rectangular matrix A 
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#


#Library Import

import numpy as np
import pandas as pd

#Script to create a random values Rectangular Matrix.
columns=4
rows=3
A = np.random.randint(10, size=(rows, columns))

print("Next is displayed the random rectangular Matrix:\n")
print(A)

    #----------------------------------------------------------------------#
    # What is the rank and trace of A?
    #----------------------------------------------------------------------#

#Rank
print("\n\n The rank of the A Matrix is:\n")
print(np.linalg.matrix_rank(A))

#Trace
print("\n\n The trace of the A Matrix is:\n")
print(np.trace(A))


    #----------------------------------------------------------------------#
    #What is the determinant of A?
    #----------------------------------------------------------------------#
try:
    detA=np.linalg.det(A)
    print("\n\n The determinant of A Matrix is:\n")
    print(detA)
except Exception as err:
    print("\n\n Oops!  It was not able to calculate the matrix determinant, it is not a square matrix")

    #----------------------------------------------------------------------#
    # Can you invert A? How?
    #----------------------------------------------------------------------#

# Calculating the inverse of the matrix
try:
    
    invA=np.linalg.inv(A)
    print("\n\n The inverse of A Matrix is:\n")
    print(invA)
except Exception as err:
    print("\n\n Oops!  it is not a square matrix, so there is not possible to calculate tje inverse matrix, but you can calculate the pseudo inverse")
    psinvA=np.linalg.pinv(A)
    print("\n\n The pseudoinverse of A Matrix is:\n")
    print(psinvA)

    #----------------------------------------------------------------------#
    # --- How are eigenvalues and eigenvectors of A’A and AA’ related? What interesting differences can you notice between both?
    #----------------------------------------------------------------------#

#Calculating A Transpose

ATransp=A.transpose()
print("A Transpose: \n",ATransp)
print("Original A Matrix: \n",A)

AAT=np.dot(A,ATransp)
ATA=np.dot(ATransp,A)

print("AA': \n",np.dot(A,ATransp))
print("A'A: \n",np.dot(ATransp,A))


#Lets call U as the eingenvectors matrix of AA' and V the eigenvectors of A'A 

#Using the numpy svd function, we can get the AA' and A'A eigenvectors

eigenvalues_AAT, U = np.linalg.eigh(np.dot(A,ATransp))
print("eigenvalues AA':",eigenvalues_AAT)
print("eigenvectors:\n",U)

eigenvalues_ATA, V = np.linalg.eigh(np.dot(ATransp,A))
print("eigenvalues A'A:",eigenvalues_ATA)
print("eigenvectors:\n",V)

"""
As we can see, the eigenvalues are the same for A'A and AA'.A

Using the numpy svd function, we can get the AA' and A'A eigenvectors and the eigenvalues. Lets call them U,V and S respectively.
Now, we can probe that A=U.S.V'

"""



U, S1, Vh = np.linalg.svd(A, full_matrices=True)


S = np.zeros((3,4))
S[:,:-1] = np.diag(S1)
S[np.isnan(S)] = 0

A2=np.dot(np.dot(U,S),Vh)

print("A2: \n",A2)
print("U,V,S",U,Vh,S)

#So we can see that A'A and AA' are related in order tha we can reconstruct the original A matrix 
# through the A'A, AA' eigenvectors and the eigenvalues.