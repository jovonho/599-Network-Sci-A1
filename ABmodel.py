
from scipy import sparse
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from A1 import get_degrees


def generate_AB_graph(n):

    graph = coo_matrix(([1,1], ([0, 1], [1, 0])),  shape=(n, n), dtype=np.int32)

    row = [0,1]
    col = [1,0]

    degrees = {
        0: 1,
        1: 1
    }

    total_degree = 2

    print(graph.todense())

    for i in range(2, n):

        # Copy the dictionary items, since we can't modify it while iterating
        degrees_at_t = list(degrees.items())
        total_degree_at_t = total_degree

        for k, k_deg in degrees_at_t:

            p = k_deg / total_degree_at_t
            
            print(f"Probability of linking node {i} to node {k}: {p}")

            make_link = np.random.choice([0,1], p = [(1-p), p])

            if make_link:
                print(f"\tNode {i} and node {k} linked!")
                row.extend([k, i])
                col.extend([i, k])

                degrees[i] = 1
                degrees[k] += 1

                total_degree += 2

    nnz = len(row)    
    graph = coo_matrix((np.ones(nnz), (row, col)),  shape=(n, n), dtype=np.int32)
    print(graph.todense())


    return graph




def main():
    graph = generate_AB_graph(n=10)


if __name__=='__main__':
    main()