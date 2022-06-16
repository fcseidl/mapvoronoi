import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2

from pathfinder import PathFinder


class Voronoi:
    """
    Create Voronoi diagrams on sophisticated maps.

    Parameters (each is optional and will be assigned a
    trivial default value if not provided):

        elevation : 2D array containing elevaton data by x and y.
                    Default is zero elevation
                    everywhere.
        mobility : 2D array with values in [0, 1] indicating
                    ease of travel as a proportion of max
                    speed, by latitude and longitude. Default is
                    all ones, equal speed everywhere. Note that
                    if mobility[x, y] == 0, a location will
                    be treated as inaccessible.
        seeds : Default is 4. If a list of x,y coordinates is
                    provided, these will be the centroids used
                    in the Voronoi decomposition. If a positive
                    integer is provided, that number of centroids
                    will be chosen automatically according to a
                    heuristic to create even spacing.
        pix_size : scalar indicating the size of a pixel, in
                    the same units as the values in elevation.
                    For instance, if a 1 kilometer square
                    region is represented by a 100 x 100 grid,
                    and elevation values are in meters, then
                    pix_size should be 1 km / 100 pix = 10 m/pix.
                    Defaults to 1.
        nbd_size : Controls charcteristic degree of the sparse
                    graph fed to dijsktra's algorithm. At the
                    default value of 8, distances calculated may
                    exceed true geodesic distance by up to 8%.
                    Passing a value of 16 will tighten this bound
                    to 3%, but potentially more than double runtime.

    Shortest paths are geodesic curves on the surface given by
    z = elevation[x, y], where local speed is mobility[x, y].
    """

    def __init__(self, elevation=None, mobility=None, seeds=4,
                 pix_size=1, nbd_size=8):
        if elevation is None:
            elevation = np.zeros_like(mobility)
        if mobility is None:
            mobility = np.ones_like(elevation)

        ns, _ = elevation.shape
        self._elev = elevation
        norm = plt.Normalize()
        self._colors = cm.terrain(norm(elevation))

        print("Building sparse graph...")
        pf = PathFinder(elevation, mobility, nbd_size)
        if type(seeds) == int:
            print("Determining %d Voronoi seeds..." % seeds)
            centroids = self._automatic_centroids(pf, seeds)
            print(centroids)
        else:
            centroids = seeds

        print("Running dijkstra's algorithm...")
        self._nearest = pf.nearest(centroids)

        # find and draw voronoi boundaries
        self._bounds = cv2.Canny(self._nearest.astype(np.uint8), 1, 1)
        self._colors[self._bounds > 0] = (0, 0, 0, 1)

        # draw each site
        for i, j in centroids:
            cv2.circle(
                img=self._colors,
                center=(j, i),
                radius=int(ns / 50),
                color=(.7, 0, .7, 1),
                thickness=-1
            )

    # Use spectral clustering to compute interesting
    # centroids from PathFinder object
    def _automatic_centroids(self, pf, n_seeds):
        # TODO: interface violation. The graph is a private pf member

        # create sparse affinity matrix and embed in R2
        affinity = pf._graph.copy()
        np.reciprocal(affinity.data, out=affinity.data)
        from scipy import sparse
        from scipy.sparse.linalg import eigsh
        lapl = sparse.csgraph.laplacian(affinity)
        _, reps = eigsh(affinity, k=n_seeds)
        reps = reps[:, 1:n_seeds+1]
        reps /= np.abs(reps).max()

        # get cluster centers of embedded representatives
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_seeds)
        km.fit(reps)

        # get vertex index the nearest representative to
        # each cluster center
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(reps)
        centroids_u = nn.kneighbors(
            km.cluster_centers_
        )[1][:, 0]

        # find pixel coordinates of vertices
        return [pf._u2ij(u) for u in centroids_u]


    def plot_flat(self):
        """
        Plot a top-down view of the Voronoi diagram
        TODO: visualize mobility somehow?
        """
        plt.imshow(self._colors)
        plt.show()

    def plot3d(self):
        """
        Plot the surface in 3D with the Voronoi diagram
        drawn onto it.
        TODO: visualize mobility somehow?
        """
        ns, ew = self._elev.shape
        X, Y = np.meshgrid(np.arange(ew), np.arange(ns))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self._elev,
                        rstride=1, cstride=1,
                        facecolors=self._colors,
                        linewidth=0, antialiased=False)
        plt.show()

def print_polygons(self):
    """
    Print the Voronoi centroids and polygons. Each centroid
    is followed by the list of vertices of its Thiessel
    Polygon.

    center1_x, center2_y

    x1, y1
    x2, y2
    ...
    xn, yn

    center2_x, center2_y

    ...
    """
    pass


if __name__ == "__main__":
    from mapgen import mapgen

    print("Generating terrain...")
    hmap = mapgen(100, 100, 102)
    #mob = (hmap > -200)      # stay out of low areas

    sites_xy = [(47, 49), (31, 14), (20, 44), (69, 44)]
    voro = Voronoi(hmap, nbd_size=16)
    voro.plot_flat()