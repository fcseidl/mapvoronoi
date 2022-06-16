from scipy import sparse
import numpy as np


class PathFinder:
    """
    Find the shortest paths in a terrain, considering
    elevation and surface quality.
    """

    def __init__(self, elevation, mobility, nbd_size=8):
        ns, ew = elevation.shape
        self._stride = ew

        data = []
        u = []
        v = []
        print()
        for i in range(ns):
            # print("Computing edge weights for row %d/%d" % (i + 1, ns))
            for j in range(ew):
                # cardinal
                for k, l in [          (i-1, j),
                             (i, j-1),           (i, j + 1),
                                       (i+1, j)            ]:
                    if 0 <= k < ns and 0 <= l < ew:
                        data.append(
                            np.sqrt(
                                1 + (elevation[i, j] - elevation[k, l]) ** 2
                            ) / (1e-6 + mobility[i, j] + mobility[j, k])
                        )
                        u.append(self._ij2u(i, j))
                        v.append(self._ij2u(k, l))
                # semi-cardinal
                for k, l in [(i - 1, j - 1), (i - 1, j + 1),
                             (i + 1, j - 1), (i + 1, j + 1)]:
                    if 0 <= k < ns and 0 <= l < ew:
                        data.append(
                            np.sqrt(
                                2 + (elevation[i, j] - elevation[k, l]) ** 2
                            ) / (1e-6 + mobility[i, j] + mobility[j, k])
                        )
                        u.append(self._ij2u(i, j))
                        v.append(self._ij2u(k, l))
                # semi-semi-cardinal
                if nbd_size == 16:
                    for k, l in [                (i - 2, j - 1), (i - 2, j + 1),
                                 (i - 1, j - 2),                                 (i - 1, j + 2),
                                 (i + 1, j - 2),                                 (i + 1, j + 2),
                                                 (i + 2, j - 1), (i + 2, j + 1)]:
                        if 0 <= k < ns and 0 <= l < ew:
                            data.append(
                                np.sqrt(
                                    5 + (elevation[i, j] - elevation[k, l]) ** 2
                                ) / (1e-6 + mobility[i, j] + mobility[j, k])
                            )
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

    def nearest(self, source_xy):
        """
        Return an array of the same shape as the heightmap
        whose x,y entry is the index of the nearest point
        in source_xy.
        """
        _, __, sources = self._dijkstra(
            source_xy,
            return_predecessors=True,
            min_only=True
        )
        return sources.reshape(-1, self._stride)

    def distances(self, source_ij):
        """
        Return a 3D array whose k,l,m entry is the distance
        from source_ij[k] to l,m in the terrain along the
        shortest path found. The units are rows/columns
        of the heightmap.
        """
        return np.reshape(
            a=self._dijkstra(source_ij),
            newshape=(len(source_ij), -1, self._stride)
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
    from matplotlib import cm

    ns = 100
    ew = 120

    # hmap = np.zeros((ns, ew))
    hmap = mapgen(ns, ew, 1)
    pf = PathFinder(hmap)
    darr = pf.distances([(50, 60)])

    X, Y = np.meshgrid(np.arange(ew), np.arange(ns))
    color = cm.gray(1 - darr[0] / darr.max())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, hmap, rstride=1, cstride=1,
                           facecolors=color,
                           linewidth=0, antialiased=False)
    plt.show()