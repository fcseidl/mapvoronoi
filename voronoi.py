import matplotlib.pyplot as plt
import numpy as np

from pathfinder import PathFinder


class Voronoi:
    """
    Create Voronoi diagrams using shortest path distance on
    height maps.
    """

    def __init__(self, heightmap, sites, colors):
        self._hmap = heightmap
        self._sites = sites

        pf = PathFinder(heightmap)
        darr = pf.distances(sites)
        self._nearest = np.argmin(darr, axis=0)
        self._colors = [[colors[n] for n in row]
                        for row in self._nearest]

    def plot_flat(self):
        """
        Plot a top-down view of the Voronoi diagram
        """
        plt.imshow(self._colors)
        plt.show()

    def plot3d(self):
        """
        Plot the surface in 3D with the Voronoi diagram
        drawn onto it.
        """
        ns, ew = self._hmap.shape
        X, Y = np.meshgrid(np.arange(ew), np.arange(ns))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, self._hmap,
                               rstride=1, cstride=1,
                               facecolors=self._colors,
                               linewidth=0, antialiased=False)
        ij = np.array(self._sites)
        ax.scatter3D(
            ij[:, 1], ij[:, 0],
            [self._hmap[s[0], s[1]] for s in ij],
            color='black', s=50
        )
        plt.show()


if __name__ == "__main__":
    from mapgen import mapgen

    hmap = mapgen(100, 100, 40)
    sites = [(50, 11), (19, 69), (79, 50)]
    colors = [(1, 0, 0, 0.2), (0, 1, 0, 0.2), (0, 0, 1, 0.2)]
    voro = Voronoi(hmap, sites, colors)
    voro.plot3d()