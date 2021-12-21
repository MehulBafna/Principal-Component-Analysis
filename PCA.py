import numpy as np

class Principal_comp_anal:
    
    def Var_Covar_matrix(self,matrix):
        self.matrix = matrix
        B = np.zeros((len(self.matrix),len(self.matrix.T)))
        for i in range(len(self.matrix.T)):
            B[:,i]=self.matrix[:,i]-np.average(self.matrix[:,i]) 
        
        M = []
        for i in range(0,len(self.matrix.T)):
            for j in range(0,len(self.matrix.T)):
                M.append(B[:,i]*B[:,j])
        
        N=[]
        for i in M:
            z = 0
            for j in i:
                z+=j
            N.append(z)
            
        C = np.array(N)
        
        P = C.reshape((len(self.matrix.T),len(self.matrix.T)))/(len(self.matrix.T)-1) # Variance-Covariance marix
        return P
    
    def Prin_anal(self,matrix):
        self.matrix = matrix 
        x = Principal_comp_anal.Var_Covar_matrix(self,matrix)
        Q,R = np.linalg.eigh(x) #Q stores eigenvalues while R store rspective eigenvectors
            
        S = np.zeros((len(R),len(R)))
        for i in range(0,len(R)):
            S[:,i]=R[:,len(R)-i-1]
                
        return S
    
if __name__=='__main__':
    x = int(input())
    y = int(input())
    L = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            L[i][j] = float(input())     
    obj = Principal_comp_anal()
    print('\n',L)
    print('\n', 'Variance-covariance matrix - ','\n',obj.Var_Covar_matrix(L))
    print('\n',obj.Prin_anal(L))
