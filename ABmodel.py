
import collections
from scipy import sparse
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import time
import logging
from logging import handlers
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

import sys
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

fh = handlers.RotatingFileHandler(filename='./log/ABmodel.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)

log.addHandler(ch)
log.addHandler(fh)


def compare_degree_distrib(A1, A2):

    degrees1 = np.array([sum(line) for line in A1.A])
    total_degree1 = sum(degrees1)

    print(f"Degrees: {degrees1}, len(degrees): {len(degrees1)}")
    print(f"Total degree: {total_degree1}")

    counts1 = collections.Counter(degrees1)

    X1, Y1 = list(counts1.keys()), list(counts1.values())

    degrees2 = np.array([sum(line) for line in A2.A])
    total_degree2 = sum(degrees2)

    print(f"Degrees: {degrees2}, len(degrees): {len(degrees2)}")
    print(f"Total degree: {total_degree2}")

    counts2 = collections.Counter(degrees2)

    X2, Y2 = list(counts2.keys()), list(counts2.values())


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))

    ax1.set_title("Method 1: random candidate choice")
    ax1.set_ylabel("P(k)")
    ax1.set_xlabel("k")
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
    ax1.tick_params(which='both', direction="in")

    ax1.scatter(X1, Y1)

    ax2.set_title("Method 2: iterating over existing nodes")
    ax2.set_ylabel("P(k)")
    ax2.set_xlabel("k")
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=10)
    ax2.tick_params(which='both', direction="in")

    ax2.scatter(X2, Y2, c='r')

    fig.tight_layout()
    plt.show()


# No binning, just raw data
def plot_degree_distrib(A):

    # degrees1 = np.array([sum(line) for line in A.A])

    degrees1 = []
    for i in range(A.shape[0]):
        row = A.getrow(i)
        row = row.toarray()[0]
        print(f"Node {i} has degree: {sum(row)}")
        degrees1.append(sum(row))


    total_degree1 = sum(degrees1)

    print(f"Degrees: {degrees1}, len(degrees): {len(degrees1)}")
    print(f"Total degree: {total_degree1}")

    counts1 = collections.Counter(degrees1)

    X1, Y1 = list(counts1.keys()), list(counts1.values())

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(13, 7))

    ax1.set_title("Method 1: random candidate choice")
    ax1.set_ylabel("P(k)")
    ax1.set_xlabel("k")
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
    ax1.tick_params(which='both', direction="in")

    ax1.scatter(X1, Y1)

    fig.tight_layout()
    plt.show()



# Complexity: O(n^2) but much quicker than randomly picking candidates in practice
def generate_AB_graph(n, m):

    # Initialize our data structures

    # These list are in COO format, where an Matrix(row[i], col[i]) = 1 if there is a link 
    # We want our graph to be undirected so we insert edges in both directions.
    row = [0,1]
    col = [1,0]

    # This dictionary will keep track of the degrees of each nodes
    nodes = {
        0: 1,
        1: 1
    }

    total_degree = 2

    for i in range(2, n):

        # Copy the current nodes and their degree
        candidate_nodes = nodes.copy()
        # Must get the keys te be able to remove values from it while iterating
        candidate_node_keys = list(candidate_nodes.keys())
        total_degree_at_t = total_degree
        links_made = 0

        log.debug(f"Adding new node {i}")
        log.debug(f"\tDegrees at time t={i}: {candidate_nodes}")

        # Here we iterate over the existing nodes always in the same order
        for k, k_deg in candidate_nodes.items():   
            
            log.debug(f"\tTrying to link node {k}") 

            # Compute the probability of linking node k to i
            p = k_deg / total_degree_at_t
            log.debug(f"\tProb of linking node {k} to {i}: {p}")

            make_link = np.random.choice([0,1], p = [(1-p), p])

            if make_link:
                links_made += 1

                log.debug(f"\t*** Nodes {k} and {i} linked!")

                # Add the new link to the row and col lists
                # We add the link in both directions (k, i) and (i, k) so the matrix will be symmetric
                row.extend([k, i])
                col.extend([i, k])

                # Increment the degrees of each node in the dictionary to use it when adding the next node
                if i in nodes:
                    nodes[i] += 1
                else:
                    nodes[i] = 1

                nodes[k] += 1

                # Total degree increases by 2, since we add symmetric links
                total_degree += 2

                if links_made == m:
                    log.debug(f"\t******All {m} links made for node {i}!!")
                    break

        if links_made < m:
            log.debug(f"\t*** No more candidate nodes. Links made for node {i}: {links_made}")

    # Data will be all ones
    nnz = len(row)    
    data = np.ones(nnz)

    # TODO BROKEN
    if False:
        with open(f"./data/AB_n{n}_m{m}.edgelist.txt", "w") as outfile:
            for i in range(0, n):
                outfile.write(f"{row[i]}\t{col[i]}\n")
            outfile.write("\n")


    # Generate the sparse matrix only at the end, from the coordinate lists
    graph = coo_matrix((data, (row, col)),  shape=(n, n), dtype=np.int32)


    return graph.tocsr()




# Complexity: O(n^2) but much quicker than randomly picking candidates in practice
def generate_AB_graph_ensure_m_edges(n, m):

    # Initialize our data structures

    # These list are in COO format, where an Matrix(row[i], col[i]) = 1 if there is a link 
    # We want our graph to be undirected so we insert edges in both directions.
    row = [0,1]
    col = [1,0]

    # This dictionary will keep track of the degrees of each nodes
    nodes = {
        0: 1,
        1: 1
    }

    total_degree = 2

    for i in range(2, n):

        # Copy the current nodes and their degree
        candidate_nodes = nodes.copy()
        # Must get the keys te be able to remove values from it while iterating
        candidate_node_keys = list(candidate_nodes.keys())
        total_degree_at_t = total_degree
        links_made = 0

        log.debug(f"Adding new node {i}")
        log.debug(f"\tDegrees at time t={i}: {candidate_nodes}")

        while links_made < m:
            log.info(f"linking node {i}")

            # Here we iterate over the existing nodes always in the same order
            for k, k_deg in candidate_nodes.items():   
                
                log.debug(f"\tTrying to link node {k}") 

                # Compute the probability of linking node k to i
                p = k_deg / total_degree_at_t
                log.debug(f"\tProb of linking node {k} to {i}: {p}")

                make_link = np.random.choice([0,1], p = [(1-p), p])

                if make_link:
                    links_made += 1

                    log.debug(f"\t*** Nodes {k} and {i} linked!")


                    # Add the new link to the row and col lists
                    # We add the link in both directions (k, i) and (i, k) so the matrix will be symmetric
                    row.extend([k, i])
                    col.extend([i, k])

                    # Increment the degrees of each node in the dictionary to use it when adding the next node
                    if i in nodes:
                        nodes[i] += 1
                    else:
                        nodes[i] = 1

                    nodes[k] += 1

                    # Total degree increases by 2, since we add symmetric links
                    total_degree += 2

                    if links_made == m:
                        log.debug(f"\t******All {m} links made for node {i}!!")
                        break

            log.debug(f"We've iterated through all the nodes. links made: {links_made} vs {m}")

            # if links_made < m:
            #     log.debug(f"\t*** No more candidate nodes. Links made for node {i}: {links_made}")

    # Data will be all ones
    nnz = len(row)    
    data = np.ones(nnz)

    row_col = [(row[i], col[i]) for i in range(len(row))]
    row_col = sorted(row_col)

    # TODO: This is broken. When we read the matrix from the edgelist its all fucked up
    if False:
        with open(f"./data/AB_ensure_n{n}_m{m}.edgelist.txt", "w") as outfile:
            for i in range(0, n):
                print(f"Writing {row_col[i][0]}\t{row_col[i][1]}\n")
                outfile.write(f"{row_col[i][0]}\t{row_col[i][1]}\n")


    # Generate the sparse matrix only at the end, from the coordinate lists
    graph = coo_matrix((data, (row, col)),  shape=(n, n), dtype=np.int32)

    return graph.tocsr()





def main():

    log.setLevel(logging.INFO)

    # log.setLevel(logging.DEBUG)

    n=1039
    m=5

    t1 = time.time()
    graph = generate_AB_graph_ensure_m_edges(n=n, m=m, save_to_file=True)
    t2 = time.time()
    print(f"Time to generate AB model(n={n}, m={m}) with version 2: {t2-t1} s")

    # compare_degree_distrib(graph1, graph2)

    print(f"Shape of graph: {graph.shape}")
    print(f"Graph nnz: {graph.nnz}")

    print(graph.A)

    A_minus_AT = graph - graph.T  

    print(f"\nIs A symmetric? (Is A - A.T all zeros?): {A_minus_AT.nnz == 0}")

    degrees = [sum(line) for line in graph]

    print(f"degrees: {degrees}")

    plot_degree_distrib(graph)


if __name__=='__main__':
    main()