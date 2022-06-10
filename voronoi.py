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
        ns, ew = heightmap.shape
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
        contours, _ = cv2.findContours(
            self._nearest,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            self._colors,
            contours,
            contourIdx=-1,
            color=(0, 0, 0),
            thickness=1
        )

        # draw each site
        for i, j in self._sites:
            pass

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
    voro = Voronoi(hmap, sites)
    voro.plot_flat()