from scipy import sparse
import numpy as np


class PathFinder:
    """
    Find the shortest paths in a 2D heightmap of a terrain.
    """

    def __init__(self, heightmap):
        ns, ew = heightmap.shape
        self._stride = ew

        data = []
        u = []
        v = []
        for i in range(ns):
            for j in range(ew):
                # diagonal
                for k, l in [(i - 1, j - 1), (i - 1, j + 1),
                             (i + 1, j - 1), (i + 1, j + 1)]:
                    if 0 <= k < ns and 0 <= l < ew:
                        data.append(
                            np.sqrt(
                                2 + (heightmap[i, j] - heightmap[k, l]) ** 2))
                        u.append(self._ij2u(i, j))
                        v.append(self._ij2u(k, l))
                # lateral
                for k, l in [          (i-1, j),
                             (i, j-1),           (i, j + 1),
                                       (i+1, j)            ]:
                    if 0 <= k < ns and 0 <= l < ew:
                        data.append(
                            np.sqrt(
                                1 + (heightmap[i, j] - heightmap[k, l]) ** 2))
                        u.append(self._ij2u(i, j))
                        v.append(self._ij2u(k, l))
        self._graph = sparse.coo_matrix((data, (u, v)))

    # Flatten heightmap coordinates.
    def _ij2u(self, i, j):
        return i * self._stride + j

    # Convert flattened coordinates back to 2D heightmap
    # coordinates.
    def _u2ij(self, u):
        return int(u / self._stride), u % self._stride

    # Internal wrapper for scipy.sparse.csgraph.dijkstra()
    # Input is in heightmap coordinates; output is
    # flattened.
    def _dijkstra(self, source_ij, **kwargs):
        ij = np.array(source_ij)
        i = ij[:, 0]
        j = ij[:, 1]
        u = self._ij2u(i, j)
        return sparse.csgraph.dijkstra(
            self._graph,
            directed=False,
            indices=u,
            **kwargs)

    def distances(self, source_ij):
        """
        Return a 3D array whose k,l,m entry is the distance
        from source_ij[k] to l,m in the terrain along the
        shortest path found. The units are rows/columns
        of the heightmap.
        """
        return np.reshape(
            a=self._dijkstra(source_ij),
            newshape=(len(source_ij), self._stride, -1)
        )

    # TODO: implement!!!
    def shortest_path(self, i_source, j_source, 
                            i_dest, j_dest):
        """
        Returns lists i_path, j_path whose kth entries are 
        i- and j-coordinates of the kth pixel on the shortest 
        path from (i_source, j_source) to (i_dest, j_dest).
        """
        pass


if __name__ == "__main__":
    from mapgen import mapgen
    import matplotlib.pyplot as plt

    hmap = mapgen(100, 120, 0)
    pf = PathFinder(hmap)
    darr = pf.distances([(50, 60)])
    print(darr)