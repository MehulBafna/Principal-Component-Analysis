'''Program for computing variance-covariance matrix and Principal Component Analysis(PCA) '''
import numpy as np

def VCM(x,y,m,n):
    A = np.random.randint(x,y,(m,n)) #Creates a random m-by-n dimensional array where each value of array is in range [x,y]
    print(A)
    B = np.zeros((m,n))
    for i in range(n):
        B[:,i]=A[:,i]-np.average(A[:,i]) 
    
    
    L = []
    for i in range(0,n):
        for j in range(0,n):
            L.append(B[:,i]*B[:,j])
    
    M=[]
    for i in L:
        z = 0
        for j in i:
            z+=j
        M.append(z)
        
    N = np.array(M)
    
    P = N.reshape((n,n))/(n-1) # Variance-Covariance marix
    Q,R = np.linalg.eigh(P) #Q stores eigenvalues while R store rspective eigenvectors
    
    S = np.zeros((len(R),len(R)))
    for i in range(0,len(R)):
        S[:,i]=R[:,len(R)-i-1]
        
    return P,S    
