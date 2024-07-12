import numpy as np
import matplotlib.pyplot as plt

class Principal_comp_anal:
    
    def Var_Covar_matrix(self, matrix):
        self.matrix = matrix
        B = np.zeros((len(self.matrix), len(self.matrix.T)))
        for i in range(len(self.matrix.T)):
            B[:, i] = self.matrix[:, i] - np.average(self.matrix[:, i]) 
        
        M = []
        for i in range(len(self.matrix.T)):
            for j in range(len(self.matrix.T)):
                M.append(B[:, i] * B[:, j])
        
        N = []
        for i in M:
            z = 0
            for j in i:
                z += j
            N.append(z)
            
        C = np.array(N)
        
        P = C.reshape((len(self.matrix.T), len(self.matrix.T))) / (len(self.matrix.T) - 1)  # Variance-Covariance matrix
        return P
    
    def Prin_anal(self, matrix):
        self.matrix = matrix 
        x = self.Var_Covar_matrix(matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(x)  # Q stores eigenvalues while R store respective eigenvectors
            
        S = np.zeros((len(eigenvectors), len(eigenvectors)))
        for i in range(len(eigenvectors)):
            S[:, i] = eigenvectors[:, len(eigenvectors) - i - 1]
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        return sorted_eigenvalues, sorted_eigenvectors

    def plot_principal_components(self, matrix):
        eigenvalues, eigenvectors = self.Prin_anal(matrix)
        
        # Project the data onto the principal components
        transformed_data = np.dot(matrix, eigenvectors)
        
        # Plotting the transformed data
        plt.figure(figsize=(10, 5))
        
        # Plot the data points along the first two principal components
        plt.subplot(1, 2, 1)
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='blue', edgecolor='k', s=50)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA - First Two Principal Components')
        
        # Plot the explained variance
        explained_variance = eigenvalues / np.sum(eigenvalues)
        plt.subplot(1, 2, 2)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center')
        plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid')
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Components')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    x = int(input('Enter number of rows: '))
    y = int(input('Enter number of columns: '))
    L = np.zeros((x, y))
    print('Enter the elements of the matrix:')
    for i in range(x):
        for j in range(y):
            L[i][j] = float(input(f'Element [{i}][{j}]: '))  
    obj = Principal_comp_anal()
    print('\nMatrix:')
    print(L)
    print('\nVariance-covariance matrix:')
    print(obj.Var_Covar_matrix(L))
    print('\nPrincipal Components (Eigenvectors):')
    eigenvalues, eigenvectors = obj.Prin_anal(L)
    print(eigenvectors)
    obj.plot_principal_components(L)
