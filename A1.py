from functools import total_ordering
from math import isnan
from os import altsep
from typing import Counter
from matplotlib import lines
import numpy as np
from numpy.core.fromnumeric import size
import scipy as sp
from scipy import sparse, linalg, stats
import pandas as pd
import matplotlib.pyplot as plt 
from numpy.polynomial import Polynomial
import time
import seaborn as sns
import math 


from memory_profiler import profile

# Complexity: O(E)
# We iterate through the edges once
def remove_self_edges(edgelist):
    orig_len = edgelist.shape[0]

    self_edge_idx = []

    for i in range(0, orig_len):
        if edgelist[i][0] == edgelist[i][1]:
            self_edge_idx.append(i)

    print(f"Self edge indices: {self_edge_idx}")

    edgelist = np.delete(edgelist, self_edge_idx, axis=0)

    print(f"{orig_len - edgelist.shape[0]} self edges removed")
    print(f"{edgelist.shape[0]} rows remain")

    return edgelist

def make_symmetric(edgelist):
    
    print(f"\nNumber edges before removing (i, j) (j, i) pairs {len(edgelist)}")

    # O(E) time, O(E) space
    # edgelist = list(set(frozenset(edge) for edge in edgelist if edge[0] != edge[1]))
    edgelist = [frozenset(edge) for edge in edgelist if edge[0] != edge[1]]

    edgelist = set(edgelist)

    edgelist = list(edgelist)

    # Complexity?
    # This will remove one of each symmetric edge of form (i, j), (j, i)
    # edgelist = list(set(edgelist))
    # print(edgelist)

    edgelist = np.array([list(edge) for edge in edgelist])

    print(f"Number edges after removing (i, j) (j, i) pairs {len(edgelist)}")

    return edgelist



def sanity_check(graph, details=False):

    A = graph
    A_minus_AT = A - A.T  

    print(f"\nIs A symmetric? (Is A - A.T all zeros?): {A_minus_AT.nnz == 0}")

    if details:
        A_minus_AT = A - A.T    

        A_plus_AT = A + A.T
        print(f"(A + A.T).nnz - (A - A.T).nnz: {A_plus_AT.nnz - A_minus_AT.nnz} (should give number of non-zero symmetric cells in A)")

        d = A_plus_AT.nnz - A_minus_AT.nnz
        print(f"A.nnz - ((A + A.T).nnz - (A - A.T).nnz): {A.nnz - d} (should give number of non-zero non-symmetric cells in A)")


def make_symmetric2(graph):
    A = graph.toarray()
    A = np.maximum(A, A.transpose())
    return sparse.csr_matrix(A)


def load_matrix(filename="./data/metabolic.edgelist.txt"):
    t1 = time.time()
    edgelist = np.loadtxt(filename, dtype=np.int32)

    # edgelist = remove_self_edges(edgelist)
    edgelist = make_symmetric(edgelist)
    
    # Create the graph in CSC form
    rows = edgelist[:,0]
    cols = edgelist[:,1]
    data = np.ones(len(edgelist))

    n = np.amax(edgelist) + 1
    graph = sparse.coo_matrix((data ,(rows, cols)), shape=(n, n), dtype=np.int32)
    graph = graph.tocsr()

    # TODO: Complexity?
    A = graph + graph.T


    print(f"A nnz: {A.nnz}")
    print(f"A shape: {A.shape}")
    
    t2 = time.time()
    print(f"Time to load the graph: {t2-t1} s")

    return A



def plot_degree_distrib(A):

    degrees = get_degrees(A)

    total_degree = sum(degrees)
    print(f"\nTotal degree: {total_degree}")

    N = A.shape[0]

    print(f"Degrees: {degrees}, len(degrees): {len(degrees)}")

    # Count the number of occurence of each degree
    counts = Counter(degrees)
    print(counts)


    k, Nk = np.array(list(counts.keys())), np.array(list(counts.values()))

    print(f"k: {k}")

    # Calculate pk by dividing the number of nodes of a degree by total number of nodes
    pk = Nk / N

    print(f"max k: {max(k)}")

    print(f"pk {pk}")


    stop = np.ceil(np.log10(max(k)))

    # This is a dark art, too many bins and the first few contain no nodes 
    num_bins = int(stop) * 15
    
    print(f"Num bins: {num_bins}")
    bins = np.logspace(start=0, stop=stop, base=10, num=num_bins, endpoint=True)

    # bins = np.linspace(start=min(k), stop=max(k), num=num_bins)

    print(f"Bins: {bins}")

    # Subtract 1 to get 0-based indices
    digitized = np.digitize(degrees, bins) - 1

    # digitized = np.digitize([10,3,56,2,3,1,1,1,1,5,7], [1,2,4,8]) - 1

    print(f"digitized: {digitized}")


    bin_counts = Counter(digitized)
    print(f"Nn_count: {bin_counts}")

    Nn = []

    for i in range(0, num_bins):
        count = bin_counts[i]
        # print(f"There are {count} nodes in bin {i}")
        Nn.append(count)


    while len(Nn) < (num_bins - 1):
        Nn.append(0)

    print(f"Nn: {Nn}")

    bn = np.array(bins[1:] - bins[:-1])

    bn = np.append(bn, [bn[-1]*2])

    print(f"Bin widths: {bn}\n")

    # From NS Book Section 4.12.3. When using log binning, p<kn> is given by 
    # Nn/bn : (Number of nodes in bin n) / (size of bin n)

    pkn = np.array(Nn) / bn
    pkn = list(pkn / N)

    # Average degree of nodes in each bin
    kn = [degrees[digitized == i].mean() for i in range(0, len(bins))]

    #clean up kn and pkn
    kn_copy = kn.copy()
    pkn_copy = pkn.copy()

    for i in range(0, len(kn_copy)):
        if math.isnan(kn_copy[i]) or kn_copy[i] == 0:
            kn.remove(kn_copy[i])
            pkn.remove(pkn_copy[i])

    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))

    ax1.set_title("Raw Data")
    ax1.set_ylabel("P(k)")
    ax1.set_xlabel("k")
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
    ax1.tick_params(which='both', direction="in")

    ax1.scatter(k, pk)

    ax2.set_title("Log Binned Data")
    ax2.set_ylabel("P(k)")
    ax2.set_xlabel("k")
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=10)
    ax2.tick_params(which='both', direction="in")

    ax2.scatter(kn, pkn)


    fit1 = Polynomial.fit(np.log10(k), np.log10(pk), deg=1)
    print(f"Fit1: {fit1.coef}")

    yfit1 = lambda x: np.power(10, fit1(np.log10(x)))
    ax1.plot(k, yfit1(k), c='r', label=f"Exp: {fit1.coef[0]}")
    ax1.legend()

    fit2 = Polynomial.fit(np.log10(kn), np.log10(pkn), deg=1)
    print(f"Fit2: {fit2.coef}")

    yfit2 = lambda x: np.power(10, fit2(np.log10(x)))
    ax2.plot(kn, yfit2(kn), c='r', label=f"Exp: {fit2.coef[0]}")
    ax2.legend()

    fig.tight_layout()
    plt.show()


def get_degrees(graph):
    _, diag = sparse.csgraph.laplacian(graph, return_diag=True)
    return diag


def get_clustering_coefs(graph, avg=False):
    t1 = time.time()
    
    A3 = graph ** 3

    deg = get_degrees(graph)

    n = A3.shape[0]

    #Note: accessing by index is slow
    # clustering_coefs = [A3[i,i] / (deg[i] * (deg[i] - 1)) if (deg[i] - 1) > 0 else 0 for i in range(n)]

    A3_diag = A3.diagonal()
    clustering_coefs = [A3_diag[i] / (deg[i] * (deg[i] - 1)) if (deg[i] - 1) > 0 else 0 for i in range(n)]

    t2 = time.time()

    print(f"Time to get clustering coefs: {t2-t1} s")

    if avg:
        return clustering_coefs, deg, sum(clustering_coefs) / sum(deg)

    return clustering_coefs, deg
    

# It does phonecalls in ~7min
# About O(n3)
def plot_shortest_paths(graph):
    t1 = time.time()
    # Saw the note in documentation about D possible conflict with directed=False 
    # but out graphs are always symmetric so should work fine.
    # shortest_paths = sparse.csgraph.shortest_path(graph, method='D', directed=False, unweighted=True )

    # TODO Find out which one is fastest
    shortest_paths = sparse.csgraph.shortest_path(graph, method='D', directed=False)

    # Can give memory error with large graphs
    shortest_paths = sparse.triu(shortest_paths)

    avg_sp = shortest_paths.sum() / shortest_paths.nnz

    print(f"Average path length: {avg_sp}")

    counts = Counter(shortest_paths.data)

    print(counts)

    print(f"List of all shortest path lengths: {shortest_paths.data}")

    t2 = time.time()
    print(f"Time to get shortest paths: {t2-t1} s")

    fig, ax = plt.subplots()

    trans = ax.get_xaxis_transform()

    ax.set_title("Shortest Path Distribution")
    ax.set_xlabel("Shortest Path Length")
    ax.set_ylabel("Count")

    avg_cc_label = "{:.2f}".format(avg_sp)
    ax.axvline(x=avg_sp, color='grey', linestyle='--', alpha=.7)
    ax.text(avg_sp + .02, .9, f"Mean = {avg_cc_label}", transform=trans)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)


    ax.hist(shortest_paths.data, color='purple')
    # sns.distplot(shortest_paths.data, hist = False, kde = True, kde_kws = {'color': 'purple', 'shade': True}, ax=ax)

    fig.tight_layout()

    plt.show()



def get_connected_compo(graph):
    t1 = time.time()
    con_comp, labels = sparse.csgraph.connected_components(graph, directed=False, return_labels=True)
    print(f"Number of Connected Components: {con_comp}")

    N = labels.shape[0]
    print(f"Length of labels (num nodes): {N}")

    counts = Counter(labels)

    GCC, GCC_len = counts.most_common(1)[0]

    nodes_in_GCC = GCC_len / N

    print(f"Proportion of nodes in GCC: {nodes_in_GCC}")

    t2 = time.time()    
    print(f"Time to get comnnected components: {t2-t1} s")



def get_degree_correl(graph):
    t1 = time.time()
    degs = get_degrees(graph)
    n = degs.shape[0]

    X = []
    Y = []

    # # O(n2)
    # # Efficient row slicing with CSR
    # for i in range(n):
    #     row = graph[i,:].toarray()[0]
    #     # print(f"Row {i}: {row}")
        
    #     for j in range(n):
    #         if row[j] == 1:
    #             # print(f"Node {i} links to node {j}")
    #             X.append(degs[i])
    #             Y.append(degs[j])


    # O(n2)
    # Efficient row slicing with CSR
    for i in range(n):
        row = graph.getrow(i).toarray()[0]
        # print(f"Row {i}: {row}")
        
        for j in range(n):
            if row[j] == 1:
                # print(f"Node {i} links to node {j}")
                X.append(degs[i])
                Y.append(degs[j])



    t2 = time.time()    
    print(f"Time to calculate degree correlations: {t2-t1} s")

    # correl = np.corrcoef(X, Y)
    # print(f"Correl {correl}")

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print(f"R2 {r_value}")
    print(f"slope {slope}")
    print(f"intercept {intercept}")


    dataset = pd.DataFrame({'di': X, 'dj': Y}, columns=['di', 'dj'])

    # Should be length 2L * 2
    print(dataset.shape)
    print(dataset)

    print(f"Max X: {max(X)}")
    print(f"Max Y: {max(Y)}")

    print(f"Size of X: {len(X)}")
    print(f"Size of Y: {len(Y)}")

    fig, ax = plt.subplots()
    ax.set_title("Degree Correlation")
    ax.set_xlabel("di")
    ax.set_ylabel("dj")

    # Create empty plot with blank marker containing the extra label
    R = "{:.2f}".format(r_value)
    ax.plot([], [], ' ', label=f"R: {R}")

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    sns.histplot(data=dataset, x="di", y="dj", discrete=(True, True), cbar=True)

    ax.legend() 
    fig.tight_layout()

    plt.show()



def b_plot_clustering_coef_distrib(A):
    cc, _, avg_cc = get_clustering_coefs(A, avg=True)

    print(f"Avg cc: {avg_cc}")

    counts = Counter(cc)

    X, Y = list(counts.keys()), list(counts.values())

    bins = np.linspace(min(X), max(X), num=100)

    y_bins = np.linspace(min(Y), max(Y), num=10)

    fig, ax = plt.subplots(figsize=(13, 7))



    ax.set_title("Metabolic: Clustering Coef Distribution")
    ax.set_xlabel("Local Clustering Coefficient")
    ax.set_ylabel("Density")

    avg_cc_label = "{:.3f}".format(avg_cc)

    # Draw line for the mean
    ax.axvline(x=avg_cc, color='grey', linestyle='--', alpha=.7)

    # Used to position the mean label
    trans = ax.get_xaxis_transform()
    ax.text(avg_cc + .02, .9, f"Mean = {avg_cc_label}", transform=trans)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)

    # ax.hist(cc, bins=bins, color='purple')
    # sns.displot(cc, kind='kde', kde_kws = {'color': 'purple', 'shade': True}, ax=ax)
    sns.kdeplot(data=cc, fill=True, color='purple', ax=ax)

    fig.tight_layout()
    plt.show()


def eigenval_distrib(graph):

    t1 = time.time()

    L = sparse.csgraph.laplacian(graph)

    # Need to use float
    eigenvals = sparse.linalg.eigsh(graph.asfptype(), k=graph.shape[0]-1, return_eigenvectors=False, which = 'SM')
    eigenvals_L = sparse.linalg.eigsh(L.asfptype(), k=L.shape[0]-1, return_eigenvectors=False, which = 'SM')

    t2 = time.time()    
    print(f"Time to gt Eigenvals: {t2-t1} s")

    print(f"Adjacency eigenvals shape: {eigenvals.shape}")
    print(f"Eigenvalues of adjacency matrix: {eigenvals}")
    
    print(f"Laplacian eigenvals shape: {eigenvals_L.shape}")
    print(f"Eigenvalues of Laplacian matrix: {eigenvals_L}")
    
    num_zeros = sparse.csgraph.connected_components(sp.sparse.csgraph.laplacian(graph), directed=False, return_labels=False)
    print(f"Spectral gap: {eigenvals_L[len(eigenvals_L)-1-num_zeros]}")

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(13, 7))

    ax1.set_title("Eigenvalue Distribution of Adjacency Matrix")
    ax1.set_xlabel("Eigenvalue")
    ax1.set_ylabel("Density")

    ax1.grid(True, which='both', linestyle='--')
    ax1.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    sns.kdeplot(data=eigenvals.real, fill=True, color='purple', ax=ax1)

    ax2.set_title("Eigenvalue Distribution of Laplacian Matrix")
    ax2.set_xlabel("Eigenvalue")
    ax2.set_ylabel("Density")   

    ax2.grid(True, which='both', linestyle='--')
    ax2.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    sns.kdeplot(data=eigenvals_L.real, fill=True, color='purple', ax=ax2)

    fig.tight_layout()
    plt.show()


# Complexity: 
def g_plot_clustering_degree_rel(A):

    cc, d = get_clustering_coefs(A)

    print(f"Size of CC: {len(cc)}")
    print(f"Size of degrees: {len(d)}")

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.set_title("Clustering Coef vs Degree")
    ax.set_xlabel("Local Clustering Coefficient")
    ax.set_ylabel("Degree")
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.grid(True, which='both', linestyle='--')
    ax.tick_params(which='both', direction="in", grid_color='grey', grid_alpha=0.2)
    # ax.scatter(cc, d, c='purple')

    dataset = pd.DataFrame({'clust_coef': cc, 'degree': d}, columns=['clust_coef', 'degree'])

    sns.histplot(data=dataset, x="clust_coef", y="degree", discrete=(True, True), cbar=True)

    fig.tight_layout()
    plt.show()



def main():
    t1 = time.time()

    print("\n\n")

    # A = load_matrix("./data/powergrid.edgelist.txt")
    A = load_matrix("./data/collaboration.edgelist.txt")
    # A = load_matrix("./data/test.txt")
    # A = generate_AB_graph(2018, 2018)
    # print(A.todense())

    plot_degree_distrib(A)

    # plot_shortest_paths(A)

    # get_connected_compo(A)

    get_degree_correl(A)

    # eigenval_distrib(A)

    # b_plot_clustering_coef_distrib(A)

    g_plot_clustering_degree_rel(A)

    t2 = time.time()
    print(f"Total running time: {t2-t1} s")

    print("\n\n\n")


if __name__=='__main__': 
    main()
