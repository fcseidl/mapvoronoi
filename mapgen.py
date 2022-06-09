import numpy as np
from perlin_noise import PerlinNoise


def mapgen(ns, ew, seed):
    """Create a height map of a virtual fractal terrain."""
    noise_map = np.zeros((ns, ew))
    for scale in range(4):
        noise = PerlinNoise(octaves=2 ** scale, seed=seed)
        layer = np.sqrt(ns**2 +ew**2) * np.array([
            [noise([lat / ns, lon / ew]) for lon in range(ew)]
            for lat in range(ns)
        ])
        noise_map += 0.5 ** scale * layer
    return noise_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    terrain = mapgen(100, 100, 3)
    X, Y = np.meshgrid(np.arange(100), np.arange(100))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, terrain, cmap='gray',
                           linewidth=0, antialiased=False)
    plt.show()
