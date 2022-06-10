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

    def __init__(self, heightmap, sites, verbose=True):
        ns, _ = heightmap.shape
        self._hmap = heightmap

        # color visualization of heightmap
        if verbose:
            print("Computing pretty colors...")
        norm = plt.Normalize()
        self._colors = cm.terrain(norm(heightmap))

        # find the nearest sites to each point
        if verbose:
            print("Building sparse graph...")
        pf = PathFinder(heightmap)
        if verbose:
            print("Running dijkstra's algorithm...")
        self._nearest = pf.nearest(sites)

        # find and draw voronoi boundaries
        can = cv2.Canny(self._nearest.astype(np.uint8), 1, 1)
        self._colors[can > 0] = (0, 0, 0, 1)

        # draw each site
        for i, j in sites:
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
        ax.plot_surface(X, Y, self._hmap,
                        rstride=1, cstride=1,
                        facecolors=self._colors,
                        linewidth=0, antialiased=False)
        plt.show()


if __name__ == "__main__":
    from mapgen import mapgen

    print("Generating terrain...")
    hmap = mapgen(1000, 1000, 42)
    sites = [(706, 119), (391, 104), (200, 444), (689, 946)]
    voro = Voronoi(hmap, sites)
    voro.plot3d()