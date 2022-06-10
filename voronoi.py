import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2

from pathfinder import PathFinder


class Voronoi:
    """
    Create Voronoi diagrams using shortest path distance on
    height maps.
    """

    def __init__(self, heightmap, sites):
        ns, _ = heightmap.shape
        self._hmap = heightmap
        self._sites = sites

        # color visualization of heightmap
        norm = plt.Normalize()
        self._colors = cm.terrain(norm(heightmap))

        # find the nearest sites to each point
        pf = PathFinder(heightmap)
        darr = pf.distances(sites)
        self._nearest = np.argmin(
            darr, axis=0
        ).astype(np.uint8)

        # find and draw voronoi boundaries
        can = cv2.Canny(self._nearest, 1, 1)
        self._colors[can > 0] = (0, 0, 0, 1)

        # draw each site
        for i, j in self._sites:
            cv2.circle(
                img=self._colors,
                center=(j, i),
                radius=int(ns / 50),
                color=(.7, 0, .7, 1),
                thickness=-1
            )

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

    hmap = mapgen(50, 50, 6)
    sites = [(25, 11), (9, 41), (22, 3)]
    voro = Voronoi(hmap, sites)
    voro.plot_flat()