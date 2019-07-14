#!/usr/bin/python
"""
WNLL interpolation.
External dependencies: pyflann, scipy

Author: Bao Wang
    Department of Mathematics, UCLA
Email: wangbaonj@gmail.com
Date: Nov 11, 2017
"""
#------------------------------------------------------------------------------
# Numpy module
#------------------------------------------------------------------------------
import numpy as np
import numpy.matlib

#------------------------------------------------------------------------------
# Pyflann module:
# For fast approximate nearest neighbor searching.
#------------------------------------------------------------------------------
import sys
sys.path.insert(0, '../pyflann')
from pyflann import *

#------------------------------------------------------------------------------
# Scipy module.
#------------------------------------------------------------------------------
import scipy.sparse
import scipy.sparse.linalg

def weight_ann(data, num_s=15, num_s_normal=8):
    """
    This function is used to compute the weight matrix.
    Input: data.
           num_s: number of nearest neighbors used.
           num_s_normal: the index of the distant used to normalize the data.
    Output: sparse weight matrix.
    """
    [m, n] = data.shape
    fea = data.T
    
    # Build a KD tree for KNN search
    flann = FLANN()
    idx, dist = flann.nn(
                         fea, fea, num_s, algorithm="kmeans",
                         branching=32, iterations=500, checks=512
                        )
    
    # Construct the sparse matrix
    row_sigma = range(n)
    col_sigma = range(n)
    diag_sigma = 1./(dist[:, num_s_normal-1] + 1.e-10)
    sigma = scipy.sparse.coo_matrix((diag_sigma, (row_sigma, col_sigma))).tocsc()
    
    dist_scipy = scipy.sparse.coo_matrix(dist.T).tocsc()
    tmp = -((dist_scipy*sigma).power(2))
    #tmp = -dist_scipy*sigma
    
    [m1, n1] = tmp.shape
    row_w = []; col_w = []; val_w = []
    
    item = tmp.nonzero()
    row_w = list(item[0]); col_w = list(item[1])
    for iter1 in range(len(row_w)):
         val_w.append(np.exp(tmp[row_w[iter1], col_w[iter1]]))
    
    w = scipy.sparse.coo_matrix((val_w, (row_w, col_w))).tocsc()
    id_row = np.matlib.repmat(np.arange(n), num_s, 1)
    id_col = idx.T
    
    size1 = max(m1, n1)
    m2, n2 = id_row.shape
    
    id_row = np.array(id_row)
    id_col = np.array(id_col)
    w = np.array(w.todense())
    print('w shape: ', w.shape)
    id_row_vector = list(np.reshape(id_row, m2*n2))
    id_col_vector = list(np.reshape(id_col, m2*n2))
    w_vector = list(np.reshape(w, m2*n2))
    for i in range(n):
         id_row_vector.append(i)
         id_col_vector.append(i)
         w_vector.append(1.0)
    
    y = scipy.sparse.coo_matrix((w_vector, (id_row_vector, id_col_vector))).tocsc()
    return y


def weight_GL(W, g, id_o, id_c, flag):
    """
    This function is used to find the ID of an instance by WNLL.
    W: weight matrix.
    g: labeled value.
    id_o: index of the labeled points.
    id_c: index of the unlabeled points.
    flag: 1: WNLL; 0: GL.
    """
    n, m = W.shape
    uf = np.zeros((n,))
    uf[id_o] = g
    u = uf[id_c]
    
    W_Laplace_full = W + W.T
    if flag is 0:
        gamma =0            #GL
    else:
        gamma = n/len(id_o) #WNLL
    
    W_Laplace = W_Laplace_full[id_c, :]
    W_Laplace = W_Laplace[:, id_c]
    
    W_Ls = W[id_o, :];
    W_Ls = W_Ls[:, id_c];
    W_Ls = W_Ls.T
    
    W_Rs = W_Laplace_full[id_c, :]
    W_Rs = W_Rs[:, id_o]
    W_RHs = (W_Rs + gamma*W_Ls)
    rhs = W_RHs * g
    
    tmpMat1 = W_Laplace_full[id_c, :]
    sumvec1 = tmpMat1.sum(axis=1)
    sumvec2 = W_Ls.sum(axis=1)
    sumvec2 = sumvec2 * gamma
    sumvec3 = sumvec1 + sumvec2

    sumvec = np.zeros((np.product(sumvec3.shape),))
    for idx in range(len(sumvec3)):
        sumvec[idx] = float(sumvec3[idx])
    
    Mat1 = scipy.sparse.diags(sumvec).tocsc()
    coef_mat = Mat1 - W_Laplace
    
    #u = scipy.sparse.linalg.spsolve(coef_mat, rhs)
    u = scipy.sparse.linalg.bicgstab(coef_mat, rhs, tol=1.e-7, maxiter=1e5)
    
    #print('Shape of u: ', u[0].shape)
    uf[id_c] = u[0]
    return uf
