
from os import link
from scipy import sparse
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from A1 import plot_degree_distrib
import time
import logging
from logging import handlers

log = logging.getLogger(__name__)

import sys
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

fh = handlers.RotatingFileHandler(filename='./log/ABmodel.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)

log.addHandler(ch)
log.addHandler(fh)


# Complexity: O(n^2)
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

        # Copy the dictionary items, since we can't modify it while iterating
        nodes_t = nodes.copy()
        node_keys_t = list(nodes_t.keys())
        total_degree_at_t = total_degree
        links_made = 0

        log.info(f"Adding new node {i}")
        log.debug(f"\tDegrees at time t={i}: {nodes_t}")

        while links_made < m:

            # Break if no more candidate nodes
            if len(node_keys_t) == 0:
                log.debug(f"\t*** No more candidate nodes. Links made for node {i}: {links_made}")
                break

            # Randomly choose a candidate node k, removing it from the list of candidates
            # Note: since i is not in the existing node, no self-links can be created
            k = np.random.choice(node_keys_t)
            node_keys_t.remove(k)

            log.debug(f"\tRandomly picked node {k}") 

            # Get the degrees of node k
            k_deg = nodes_t[k]

            # Compute the probability of linking node k to i
            p = k_deg / total_degree_at_t
            log.debug(f"\tProb of linking node {k} to {i}: {p}")

            make_link = np.random.choice([0,1], p = [(1-p), p])

            if make_link:
                links_made += 1

                log.debug(f"\t*** Nodes {k} and {i} linked!")

                if links_made == m:
                    log.debug(f"\t******All {m} links made for node {i}!!")

                # Add the new link to the row and col lists
                # We add the link in both directions (k, i) and (i, k) so the matrix will be symmetric
                row.extend([k, i])
                col.extend([i, k])

                # Increment the degrees of each node in the dictionary to use it when adding the next node
                nodes[i] = 1
                nodes[k] += 1

                # Total degree increases by 2, since we add symmetric links
                total_degree += 2


    # Data will be all ones
    nnz = len(row)    
    data = np.ones(nnz)

    # Generate the sparse matrix only at the end, from the coordinate lists
    graph = coo_matrix((data, (row, col)),  shape=(n, n), dtype=np.int32)

    return graph.tocsr()




def main():

    log.setLevel(logging.INFO)

    # UNCOMMENT for more logging    
    # log.setLevel(logging.DEBUG)

    t1 = time.time()

    n=1000
    m=2

    graph = generate_AB_graph(n=n, m=m)
    t2 = time.time()
    print(f"Time to generate AB model with n={n}, m={m}: {t2-t1} s")

    plot_degree_distrib(graph)


if __name__=='__main__':
    main()