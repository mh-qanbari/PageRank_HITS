import numpy as np
from IndexMap import IndexMap
from PageRank import PageRank
from HITS import HITS

nan = float("nan")

if __name__ == '__main__':
    N = 7115
    E = 103689
    A_t = np.zeros((N, N), dtype=np.float)
    A = np.zeros((N, N), dtype=np.float)
    index_map = IndexMap()
    with open("Wiki-Vote.txt", mode='r') as data_file:
        for line in data_file:
            if line[0] == '#':
                continue
            line = line.replace('\n', '')
            ij_list = line.split('\t')
            i = index_map.getIndex(int(ij_list[0]))
            j = index_map.getIndex(int(ij_list[1]))
            A_t[j, i] = 1
            A[i, j] = 1

    print " [1] : PageRank ......................................... "
    pr = PageRank(beta=0.8, max_err=0.0001)
    pr.initTransposedMat(A_t)
    pr.normalize()
    iter_count = pr.run()
    print "Converged on", iter_count, "iterations"
    indexes = pr.v.argsort()[-10:][::-1]
    print "Best nodes:"
    for index in indexes:
        print '\t', index_map.nodes[index], '\t', pr.v[index]
    print " [1]; ................................................... "
    print
    print " [2] : HITS ............................................. "
    hits = HITS(max_err=0.0001)
    hits.initMat(mat=A, mat_t=A_t)
    # hits.normalize()
    iter_count = hits.run()
    print "Converged on", iter_count, "iterations"
    h_indexes = hits.h.argsort()[-10:][::-1]
    print "Best Hubbs:"
    for index in h_indexes:
        print '\t', index_map.nodes[index], '\t', hits.h[index]
    a_indexes = hits.a.argsort()[-10:][::-1]
    print "Best Auths:"
    for index in a_indexes:
        print '\t', index_map.nodes[index], '\t', hits.a[index]
    print " [2]; ................................................... "
