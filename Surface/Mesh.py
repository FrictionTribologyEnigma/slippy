import numpy as np
from slippy.surface import Surface
import math

__all__=['Mesh']

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
        Parameters required for the selected method
    depth : int optional (10)
        The depth of the mesh in the same units as the grid spacing and the 
        height infromation
    nodes : 3 by N numpy array optional (None)
        Array of nodes in the mesh (x,y,z coordinates for each node) makes 
        functionallity avalible to externally gererated meshes if this is 
        provided elements must also be provided
    elements : 8 by E numpy array optional (None)
        Array of elements in the mesh, with node numbers for each if this is 
        set nodes must also be set
    
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
    nodes=np.array([])
    elements=np.array([])
    _is_quadratic=False
    profile=np.array([])
    
    def __init__(self, surface : Surface = None, grid_spacing : float = None,
                 method : str = 'full', parameters : dict = None, 
                 depth : float = None, nodes : np.Array = None, 
                 elements : np.Array = None):
        do_mesh=True #flag to say if it needs meshing
        if isinstance(surface, Surface): #surface is surface
            self.profile=surface.profile
            if grid_spacing is None:
                if surface.grid_spacing is None:
                    raise ValueError("Grid spacing not set, mesh failed")
                else:
                    self.grid_spacing=surface.grid_spacing
            else:
                self.grid_spacing=float(grid_spacing)
        elif surface is not None: # try surface being array like
            try:
                self.profile=np.array(surface, dtype=float)
            except ValueError:
                raise ValueError("surface must be a surface object or"
                                 " convertable to a float array, mesh failed")
        elif elements is not None and nodes is not None:
            do_mesh=False #dosen't need meshing if mesh is given
            self.nodes=np.array(nodes, dtype=float)
            self.elements=np.array(elements, dtype=int)
            if self.nodes.shape[0]!=3:
                raise ValueError("Nodes array is wrong shape, required 3 by n," 
                                 " found {}".format(self.nodes.shape))
            if self.elements.shape[0]!=8:
                raise ValueError("Elements array is wrong shape, required 8 by"
                                 " n, found {}".format(self.elements.shape))
        else:
            raise ValueError("Invalid combination of parameters set, "
                             "mesh failed")
        if do_mesh:
            if depth is None or not parameters:
                raise ValueError("Depth and parameters must be provided for "
                                 "meshing")
            if method=="full":
                parameters=self._full(parameters, depth)
            elif method=="reducing":
                parameters=self._reducing(parameters, depth)
            else:
                raise ValueError("Unrecognised mesh method {}".format(method))
            if parameters:
                keys=' '.join(list(parameters.keys()))
                raise ValueError("Unrecognised keys "
                                 "'{}' in parameters".format(keys))
        
    def _full(self, parameters, depth):
        """
        valid keys in parameters:
            max_aspect
            min_aspect
            mode {linear, exponential}
        """
        valid_modes=['linear', 'exponential']
        aspect=[1,6]
        mode='linear'
        if depth is None:
            raise ValueError("Depth must be set, meshing failed")
        
        if 'max_aspect' in parameters:
            aspect[1]=parameters.pop('max_aspect')
        if 'min_aspect' in parameters:
            aspect[0]=parameters.pop('min_aspect')
        if 'mode' in parameters:
            mode=parameters.pop('mode')
            if not mode in valid_modes:
                raise ValueError("Unrecognised mode : "
                                 "{}, mesh failed".format(mode))
        
        h_min=min(self.profile.flatten())
        self.profile=self.profile-h_min+depth
        
        h_min=depth
        h_max=max(self.profile.flatten())
        
        if aspect[0]>aspect[1]:
            seg_max_top=aspect[0]*self.grid_spacing
            seg_max_bottom=aspect[1]*self.grid_spacing/h_min*h_max
        else:
            seg_max_top=aspect[0]*self.grid_spacing/h_min*h_max
            seg_max_bottom=aspect[1]*self.grid_spacing
        
        n_segs=math.ceil(h_max/(seg_max_top+seg_max_bottom)/2)
        segs_raw=np.cumsum(np.linspace(seg_max_bottom, seg_max_top, n_segs))
        
        # normalised y coodinates for each node multiply by the hieght of the 
        # surface to get the actual y coordinates
        segs_norm=np.insert(segs_raw, 0, 0)/h_max 
        
        n_pts=self.profile.size
        
        #just index X and Y using mod operator
        X, Y = np.meshgrid(np.arange(self.profile.shape[0])*self.grid_spacing,
                           np.arange(self.profile.shape[1])*self.grid_spacing)
        Z=np.reshape(np.repeat(segs_norm, n_pts, (n_segs, 
          self.profile.shape[0], self.profile.shape[1])))*self.profile
        
        # got all the nodes now construct the elements
        
    
    def merge_points(self, distance):
        pass
        
    def save(self, file, output_type):
        pass
    
    def make_quadratic():
        #make set of edges
        #add node with 'number' half way between the two nodes for each edge
        pass
        
        
        
        
        
        
        
        