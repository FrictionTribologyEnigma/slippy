import numpy as np
from slippy.surface import Surface
import math
import itertools

__all__ = ['Mesh']


class Mesh(object):
    """
    A class for containing, saving and manipulating meshes of surfaces
    
    Keyword parameters
    ------------------
    surface : Surface or array-like optional (None)
        A surface object or array of surface heights if an array is supplied 
        the grid_spacing keyword must also be set, if a surface object is given
        the grid_spacing property of the surface must be set. If neither are
        given the nodes and elements keywords must be set.
    grid_spacing : float optional (None)
        The spacing between grid points, required if an array is given or if 
        the grid_spacing property is not set on the surface object
    method : str {'full', 'reducing'} optional ('full')
        The method used to create the mesh, see notes
    parameters : dict optional (None)
        Parameters required for the selected method, see notes
    depth : int optional (10)
        The depth of the mesh in the same units as the grid spacing and the 
        height infromation
    nodes : 3 by N numpy array optional (None)
        Array of nodes in the mesh (x,y,z coordinates for each node) makes 
        functionallity avalible to externally gererated meshes if this is 
        provided elements must also be provided
    elements : Dict
        With keys: hex_elements, wedge_elements, tet_elements, pyramid_elements
        each of which should be an array of node numbers realting to each
        element of the relevent type 
    
    Attributes
    ----------
    nodes
    elements
    
    Methods
    -------
    make_quadratic
    merge_points
    write
    
    See Also
    --------
    
    
    Notes
    -----
    """
    nodes = None
    hex_elements = None
    wedge_elements = None
    pyramid_elements = None
    tet_elements = None
    _is_quadratic = False
    profile = None

    def __init__(self, surface: Surface = None, grid_spacing: float = None,
                 method: str = 'full', parameters: dict = None,
                 depth: float = None, nodes: np.array = None,
                 elements: np.array = None):
        do_mesh = True  # flag to say if it needs meshing
        if isinstance(surface, Surface):  # surface is surface
            self.profile = surface.profile
            if grid_spacing is None:
                if surface.grid_spacing is None:
                    raise ValueError("Grid spacing not set, mesh failed")
                else:
                    self.grid_spacing = surface.grid_spacing
            else:
                self.grid_spacing = float(grid_spacing)
        elif surface is not None:  # try surface being array like
            try:
                self.profile = np.array(surface, dtype=float)
            except ValueError:
                raise ValueError("surface must be a surface object or"
                                 " convertable to a float array, mesh failed")
        elif elements is not None and nodes is not None:
            do_mesh = False  # dosen't need meshing if mesh is given
            self.nodes = np.array(nodes, dtype=float)
            self.elements = np.array(elements, dtype=int)
            if self.nodes.shape[0] != 3:
                raise ValueError("Nodes array is wrong shape, required 3 by n,"
                                 " found {}".format(self.nodes.shape))
            if self.elements.shape[0] != 8:
                raise ValueError("Elements array is wrong shape, required 8 by"
                                 " n, found {}".format(self.elements.shape))
        else:
            raise ValueError("Invalid combination of parameters set, "
                             "mesh failed")
        if do_mesh:
            if depth is None or not parameters:
                raise ValueError("Depth and parameters must be provided for "
                                 "meshing")
            if method == "full":
                nodes, elements, parameters = self._full(parameters, depth)
            elif method == "reducing":
                nodes, elements, parameters = self._reducing(parameters, depth)
            elif method == "gmsh":
                nodes, elements, parameters = self._gmsh(parameters, depth)
            else:
                raise ValueError("Unrecognised mesh method {}".format(method))
            if parameters:
                keys = ' '.join(list(parameters.keys()))
                raise ValueError("Unrecognised keys "
                                 "'{}' in parameters".format(keys))

    def _full(self, parameters, depth):
        """
        valid keys in parameters:
            max_aspect
            min_aspect
            mode {linear, exponential}
        """
        valid_modes = ['linear', 'exponential']
        aspect = [1, 6]
        mode = 'linear'
        if depth is None:
            raise ValueError("Depth must be set, meshing failed")

        if 'max_aspect' in parameters:
            aspect[1] = parameters.pop('max_aspect')
        if 'min_aspect' in parameters:
            aspect[0] = parameters.pop('min_aspect')
        if 'mode' in parameters:
            mode = parameters.pop('mode')
            if not mode in valid_modes:
                raise ValueError("Unrecognised mode : "
                                 "{}, mesh failed".format(mode))

        h_min = min(self.profile.flatten())
        self.profile = self.profile - h_min + depth

        h_min = depth
        h_max = max(self.profile.flatten())

        if aspect[0] > aspect[1]:
            seg_max_top = aspect[0] * self.grid_spacing
            seg_max_bottom = aspect[1] * self.grid_spacing / h_min * h_max
        else:
            seg_max_top = aspect[0] * self.grid_spacing / h_min * h_max
            seg_max_bottom = aspect[1] * self.grid_spacing

        n_segs = math.ceil(h_max / (seg_max_top + seg_max_bottom) / 2)
        segs_raw = np.cumsum(np.linspace(seg_max_bottom, seg_max_top, n_segs))

        # normalised y coodinates for each node multiply by the hieght of the 
        # surface to get the actual y coordinates
        segs_norm = np.insert(segs_raw, 0, 0) / h_max

        n_pts = self.profile.size

        # Just repete over X and Y
        X, Y = np.meshgrid(np.arange(self.profile.shape[0]) * self.grid_spacing,
                           np.arange(self.profile.shape[1]) * self.grid_spacing)
        Z = np.reshape(np.repeat(segs_norm, n_pts, (n_segs,
                                                    self.profile.shape[0], self.profile.shape[1]))) * self.profile

        node_shape = list(X.shape).append(n_pts)

        nodes = zip(itertools.cycle(X.flatten()), itertools.cycle(Y.flatten()),
                    Z)
        self.nodes = nodes

        el_nums = np.array(range(np.prod([e - 1 for e in node_shape])))
        first_nodes = (el_nums + el_nums // (node_shape[0] - 1) +
                       node_shape[0] * (el_nums // ((node_shape[0] - 1) * (node_shape[1] - 1))))
        plane_nodes = node_shape[0] * node_shape[1]
        elements = zip([first_nodes,
                        1 + first_nodes,
                        node_shape[0] + first_nodes + 1,
                        node_shape + first_nodes,
                        first_nodes + plane_nodes,
                        1 + first_nodes + plane_nodes,
                        node_shape[0] + first_nodes + 1 + plane_nodes,
                        node_shape + first_nodes + plane_nodes])
        self.hex_elements = elements

    def _reducing(self, parameters, depth):
        """
        start bottom up, mesh all the planes then work out the height of all the z points.... how though, same as previous you want rough doubling every time 
        """

        pass

    def _gmsh(self, parameters, depth):
        import gmsh

        pass

    def merge_points(self, distance):
        pass

    def save(self, output_type, file_or_filename=None):
        """
        #TODO docs
        """
        if type(file_or_filename) is str:
            file = open(file_or_filename, 'wb')
        else:
            file = file_or_filename

        if output_type == 'vtk':
            self._write_vtk(file)
        elif output_type == 'inp':
            self._write_inp(file)

        if type(file_or_filename) is str:
            file.close()

    def _write_inp(file):
        pass

    def _write_vtk(file):
        pass

    def make_quadratic(self):
        # make set of edges
        # add node with 'number' half way between the two nodes for each edge, but thats not always unique....
        pass
