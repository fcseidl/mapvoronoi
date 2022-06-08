from scipy import sparse


class PathFinder:
    """
    Find shortest paths in a 2D heightmap.
    """

    def __init__(self, map):
        pass

    def dist_matrix(self, i_list, j_list):
        """
        Return a matrix with shortest path lengths between 
        pixels at positions i_list[k], j_list[k].
        """
        pass

    def shortest_path(self, i_source, j_source, 
                            i_dest, j_dest):
        """
        Returns lists i_path, j_path whose kth entries are 
        i- and j-coordinates of the kth pixel on the shortest 
        path from (i_source, j_source) to (i_dest, j_dest).
        """
        pass