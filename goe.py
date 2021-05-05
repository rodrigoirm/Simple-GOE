import numpy as np
import matplotlib.pyplot as plt


def create_matrices (total, N):
    """Generates an array of square random matrices whose entries are independent gaussian random variables with zero mean and unit std deviation.

    Parameters:
    total (int): total number of matrices.
    N (int): sets the dimension of each matrix as NxN.

    Returns:
    np.array(matrices_list): an array of length "total" whose entries are NxN gaussian matrices.

   """
    matrices_list = []  
    for k in range(total):
        matrices_list.append(np.random.normal(0, 1, size=(N,N)))
    return np.array(matrices_list)

def GOE (gens):
    """Turns an ensemble of gaussian random matrices into an Gaussian Orthogonal Ensemble.

    Parameters:
    ensemble (ndarray): an ndarray of random matrices whose shape is (total # of matrices, N, N)

    Returns:
    ensemble (ndarray): an ndarray of symmetric gaussian matrices whose shape is (total # of matrices, N, N).

   """
    for k in range(gens.shape[0]):
        gens[k] = ( gens[k] + np.transpose(gens[k]) )/2
    return gens

def eigenval_goe (goe):
    """For each matrix in a Gaussian Orthogonal Ensemble, computes and lists the eigenvalues of said matrix in increasing order.

    Parameters:
    goe (ndarray): an ndarray of symmetric random matrices whose shape is (total of matrices, N, N).

    Returns:
    eigenvals: a list whose elements are the lists of eigenvalues of each matrix of goe.

   """
    eigenvals = []
    for k in range(goe.shape[0]):
        eigenval_k = list(np.linalg.eigvals(goe[k]))   #unsorted list of eigenvalues of the k-th matrix
        sorted_eigvals = sorted(eigenval_k) 
        eigenvals.append(sorted_eigvals)
    return eigenvals

def average (L):
    """Returns the arithmetic average of a non-empty list of floating-point numbers.

    Parameters:
    L (list): a list non-empty list of floating numbers.

    Returns:
    average(L) (float): arithmetic average of elements of L.

    """
    average = sum(L)/len(L) 
    return average

def eigendiff (eigenlist, N):
    """Computes the difference between central eigenvalues of matrices of a GOE ensemble, divides these differences by their average and lists the result.

    Parameters:
    eigenlist (list): a list with the eigenvalues of an even number of gaussian orthogonal matrices.
    N (integer): sets the dimension of the matrices.

    Returns:
    eigendiff (list): a list of the difference between central eigenvalues divided the average difference.

    """
    eigendiff = []
    for k,l in enumerate(eigenlist):
        eigendiff.append(eigenlist[k][N//2] - eigenlist[k][N//2-1]) #Remember: Python starts counting at zero.
    eigen_avrg = average(eigendiff)
    eigendiff[:] =  [diff/eigen_avrg for diff in eigendiff]
    return eigendiff

def diffhist (points,farbe):
    count, bins, ignored = plt.hist(points, 1000, density=True, color=farbe)
    plt.plot(bins, np.pi * (bins/2) *
               np.exp( - np.pi * (bins/2)**2 ),
         linewidth=2, color='k')
    return plt.show()

gens = create_matrices(15000,2)
ensemble = GOE(gens)
eigenvalues = eigenval_goe(ensemble)
goe_diff = eigendiff(eigenvalues, 2)
diffhist(goe_diff,'c')

gens2 = create_matrices(15000,4)
ensemble2 = GOE(gens2)
eigenvalues2 = eigenval_goe(ensemble2)
goe_diff2 = eigendiff(eigenvalues2, 4)
diffhist(goe_diff2,'m')

gens3 = create_matrices(15000,10)
ensemble3 = GOE(gens3)
eigenvalues3 = eigenval_goe(ensemble3)
goe_diff3 = eigendiff(eigenvalues3, 10)
diffhist(goe_diff3,'b')