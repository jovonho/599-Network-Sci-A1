import numpy as np
import scipy as sp
from scipy import sparse, linalg
import pandas as pd

def remove_self_edges(edgelist):
    # Remove self edges
    orig_len = edgelist.shape[0]

    self_edge_idx = []

    for i in range(0, len(edgelist)):
        edge = edgelist[i]

        if edge[0] == edge[1]:
            self_edge_idx.append(i)

    print(self_edge_idx)

    edgelist = np.delete(edgelist, self_edge_idx, axis=0)

    print(f"{orig_len - edgelist.shape[0]} duplicate edges removed")
    print(f"{edgelist.shape[0]} rows remain")

    return edgelist

def make_symmetric(A):
    
    for i in range(A):
        pass



def load_matrix(filename="./data/metabolic.edgelist.txt"):

    edgelist = np.loadtxt(filename, dtype=int)
    edgelist2 = edgelist

    edgelist = remove_self_edges(edgelist)

    # Ensure all rows are unique (AFAIK they already are)
    edgelist = np.unique(edgelist, axis=0)

    edgelist2 = (tuple(x) for x in edgelist2)


    




    
    # # Get rows and
    # print(edgelist[:,0], edgelist[:,1])
    rows = edgelist[:,0]
    cols = edgelist[:,1]

    # make symmetric

    # print(f"data: {len(data)}, rows: {len(rows)}, cols: {len(cols)}")
    
    data = np.ones(len(edgelist))


    n = np.amax(edgelist) + 1
    print(f"n = {n}")

    graph = sparse.coo_matrix((data ,(rows,cols)), shape=(n, n))
    A = graph
    print(repr(A))
    print(repr(sparse.tril(A)))

    # Count the number of symmetric and non-sym entries
    for i in rows:
        for j in cols:
            pass

    A_minus_AT = A - A.T

    A_plus_AT = A + A.T

    A_eq_AT = A == A.T

    print(f"A NNZ: {A.nnz}")

    A_lower = sparse.tril(A)

    # A_lower has same shape as A
    print(f"A_lower is all zeros? (Means that A was upper triangular) {A_lower.nnz == 0}")

    exit()


    # # Will return a False for every non-symmetric cell
    # print(f"A_eq_AT Number of false: {(A_eq_AT == False).sum()} (should give number of non-symmetric cells)")

    # print(f"sum(A_eq_AT == True): {(A_eq_AT == True).sum()} (should give number of symmetric cells)")
    # print(f"Total_cells - A_eq_AT NNZ: {n*n - A_eq_AT.nnz} (should give number of non-symmetric cells)")

    # Will be bigger than A.nnz if A is not symmetric
    print(f"A_plus_AT.nzz = {A_plus_AT.nnz}")

    d = A_plus_AT.nnz - A_minus_AT.nnz
    print(f"(A + A.T).nnz - (A - A.T).nnz: {A_plus_AT.nnz - A_minus_AT.nnz} (should give number of non-zero symmetric cells in A)")


    print(f"A.nnz - ((A + A.T).nnz - (A - A.T).nnz): {A.nnz - d} (should give number of non-zero non-symmetric cells in A)")


    A = A.tocsc()

    n = A.shape[0]

    num_sym = 0
    num_not_sym = 0

    for i in range(n):
        for j in range(n):
            if A[i,j] == A[j,i]:
                if A[i,j] == 1:
                    num_sym += 1
            else:
                num_not_sym += 1

    print(f"Num (non-zero) sym cells: {num_sym}")

    
    print(f"Num non_sym cells: {num_not_sym}")






    exit()

    #A_plus_AT should be symmetric
    # print(f"A_plus_AT is symmetric (Should be True)? = {((A_plus_AT == A_plus_AT.T) == True).sum() == n*n}")

    print(f"A_minus_AT NNZ: {A_minus_AT.nnz}")

    #Return the indices and values of the nonzero elements of a matrix
    i,j,v = sparse.find(A)


    A_sym = (A+A.T)/2
    print(f"A_sym.nnz: {A_sym.nnz}")

    print((A_sym.data < 1).sum())

    # Will have an nnz for each non-symmetric value in A
    # Should be 0 if completely symmetric
    Asym_min_AsymT = A_sym - A_sym.T
    print(f"(Asym - Asym.T).nzz = {Asym_min_AsymT.nnz}")

    # Should be higher than A.nnz is not symmetric, equal if symmetric
    # Should be 2L if symmetric
    Asym_plus_AsymT = A_sym + A_sym.T
    print(f"(Asym + Asym.T).nzz = {Asym_plus_AsymT.nnz}")


    sym_check_res = np.all(np.abs(Asym_min_AsymT.data) == 0 )  # tune this value

    print(sym_check_res)


    # If the number of non-zero values is equal to the number of edges
    # and we've removed the self-edges, the matrix is not symmetric.
    # If it was, nnz would be 2L
    print(f"A NNZ: {A.nnz}")
    print(f"A_minus_AT NNZ: {A_minus_AT.nnz}")

    # The sum of A will be == to NNZ since they are all 1
    print(f"sum(A) = {A.sum()}")
    print(f"sum(A_minus_AT) = {A_minus_AT.sum()}")











if __name__=='__main__': 
    load_matrix()