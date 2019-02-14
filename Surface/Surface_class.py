#mesh function

#documentation

#make list of all functionallity


import os
import numpy as np
import warnings
from math import ceil
import scipy.signal
from scipy.io import loadmat
from .ACF_class import ACF
from .roughness_funcs import roughness, subtract_polynomial, find_summits
from .roughness_funcs import get_mat_vr, get_summit_curvatures
from .roughness_funcs import get_height_of_mat_vr, low_pass_filter

__all__=['Surface', 'assurface', 'read_surface']

def assurface(profile, grid_spacing=None):
    """ make a surface from a profile
    
    Parameters
    ----------
    profile : array-like
        The surface profile
    grid_spacing : float optional (None)
        The spacing between grid points on the surface
    
    Returns
    -------
    S : Surface object
        A surface object with the specified profile and grid size
    
    See Also
    --------
    Surface
    read_surface
    
    Notes
    -----
    
    
    Examples
    --------
    
    >>> profile=np.random.normal(size=[10,10])
    >>> my_surface=assurface(profile, 0.1)
    >>> my_surface.extent
    [1,1]
    
    """
    return Surface(profile=profile, grid_spacing=grid_spacing)

def read_surface(file_name,**kwargs):
    """ Read a surface from a file
    
    Parameters
    ----------
    file_name : str
        The full path to the data file
    
    Other Parameters
    ----------------
    
    delim : str optional (',')
        The delimiter used in the data file, only needed for csv or txt files
    p_name : str optional ('profile')
        The name of the variable containing the profile data, needed if a .mat 
        file is given
    gs_name : str optional ('grid_spacing')
        The name of the variable containing the grid_spacing, needed if a .mat 
        file is given
    
    Returns
    -------
    A surface object generated from the file
    
    See Also
    --------
    Surface
    alicona_read
    scipy.io.loadmat
    
    Notes
    -----
    This function directly invokes the surface class, any other keywords that 
    can be passed to that class can be passed to this function
    
    Examples
    --------
    
    >>> # Read a csv file with tab delimiters
    >>> my_surface=read_surface('data.csv', delim='\t')
    
    >>> # Read a .al3d file
    >>> my_surface=read_surface('data.al3d')
    
    >>> # Read a .mat file with variables called prof and gs
    >>> my_surface=read_surface('data.mat', p_name='prof', gs_name='gs')
    
    """
    return Surface(file_name=file_name, **kwargs)
    

class Surface(object):
    """ Object for reading, manipulating and plotting surfaces
    
    The Surface class contains methods for setting properties, 
    examining measures of roughness and descriptions of surfaces, plotting,
    fixing and editing surfaces.
    
    Other Parameters
    ----------------
    profile : array-like optional (None)
        The surface profile
    file_name : str optional (None)
        The file name including the extention of file, for vaild types see 
        notes
    delim : str optional (',')
        The delimitor used in the file, only needed for csv or txt files
    profile : array-like optional (np.array([]))
        A surface profile 
    extent : 2 element list optional (None)
        The size of the surface in each dimention
    grid_spacing : float optional (None)
        The distance between nodes on the grid, 
    dimentions : int optional (2)
        The number of diimentions of the surface
    
    Attributes
    ----------
    profile : array
        The height infromation of the surface
        
    shape : tuple
        The numbeer of points in each direction of the profile
        
    size : int
        The total number of points in the profile
    
    grid_spacing: float
        The distance between adjacent points in the profile
        
    extent : list
        The size of the profile in the same units as the grid spacing is set in
        
    dimentions : int {1, 2}
        The number of dimentions of the surface
    
    surface_type : str {'Experimental', 'Analytical', 'Random'}
        A description of the surface type    
    
    acf, psd, fft : array or None
        The acf, psd and fft of the surface set by the get_acf get_psd and
        get_fft methods
    
    
    
    Methods
    -------
    
    show
    subtract_polynomial
    roughness
    get_mat_vr
    get_height_of_mat_vr
    find_summits
    get_summit_curvatures
    low_pass_filter
    read_from_file
    fill_holes
    resample
    get_fft
    get_acf
    get_psd
    
    See Also
    --------
    ACF
    roughness
    
    Notes
    -----
    Roughness functions are aliased from the functions provided in the surface 
    module
    
    Examples
    --------
    
    
    """
    # The surface class for descrete surfaces (typically experiemntal)
    is_descrete=False
    """ A bool flag, true if there is a profile present """
    acf=None
    """ The autocorelation function of the surface profile """
    aacf=None
    psd=None
    """ The power spectral density of the surface """ 
    fft=None
    """ The fast fourier transform of the surface """
    surface_type="Experimental"
    """ A description of the surface type """
    dimentions=2
    """ The number of spartial dimentions that """
    size=None
    
    _profile=None
    _grid_spacing=None
    _shape=None
    _extent=None
    _inter_func=None
    _allowed_keys=[]
    _mask=None
    
    
    def __init__(self,**kwargs):
        # check for file
        if 'file_name' in kwargs:
            ext=os.path.splitext(kwargs['file_name'])
            file_name=kwargs.pop('file_name')
            if ext=='pkl':
                self.load_from_file(file_name)
            else:
                kwargs=self.read_from_file(file_name, **kwargs)
        
        elif 'profile' in kwargs: # check for profile
            self.profile=kwargs['profile']
        
        kwargs=self._init_checks(kwargs) # check for other kwargs profile is set
        
        # at this point everyone should have taken what they needed
        if kwargs:
            raise ValueError("Unrecognised keys in keywords: "+str(kwargs))
        
    def _init_checks(self, kwargs={}):
        # add anything you want to run for all surface types here
        if kwargs:
            #load allowed keys straight in to attributes
            allowed_keys=self._allowed_keys
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            for key in allowed_keys:
                if key in kwargs:
                    del kwargs[key]
            
            if 'profile' in kwargs:
                self.profile=kwargs.pop('profile')
                self.is_descrete=True
                p_set=True
            else:
                p_set=False
            
            if 'grid_spacing' in kwargs:
                self.grid_spacing=kwargs.pop('grid_spacing')
                gs_set=True
            else:
                gs_set=False
            
            if 'extent' in kwargs:
                if gs_set and p_set:
                    msg=('Only grid_spacing or extent can be set with a '
                        'specific profile, it is recomended to only set grid_'
                        'spacing')
                    raise ValueError(msg)
                if p_set:
                    msg=('extent set with profile, only the first '
                        'dimention of the surface will be used to calculate '
                        'the grid spacing')
                    warnings.warn(msg)
                self.extent=kwargs.pop('extent')
            
        return kwargs
    
    @property
    def mask(self):
        """A mask used to exclude some values from analysis
        
        Either a boolean array of size self.size or a float of the value to
        be excluded
        """
        return self._mask
    
    @mask.setter
    def mask(self, value):
        if value is None:
            self._mask=None
            
        elif type(value) is float:
            if np.isnan(value):
                mask=~np.isnan(self.profile)
            else:
                mask=~self.profile==value
        else:
            mask=np.asarray(mask, dtype=bool)
            if not mask.shape==self.shape:
                msg=("profile and mask shapes do not match: profile is"
                    "{profile.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)
        
    @mask.deleter
    def mask(self):
        self._mask=None
    
    @property
    def extent(self):
        """ The extent of the surface in the same units as grid spacing
        """
        return self._extent
    
    @extent.setter
    def extent(self, value):
        """ Changes the global size of the surface
        
        Sets the global size of the surface without reinterpolation, keeps all
        other dimnetions up to date
        
        Parameters
        ----------
        extent : 2 element list
            The global size to be set in length units
            
        Returns
        -------
        
        Nothing
        
        See Also
        --------
        resample - changes the grid spacing with reinterpolation
        set_grid_spacing
        
        Notes
        -----
        
        This method should be used over editing the properties directly
        
        Examples
        --------
        
        """
        if type(value) is not list:
            msg="Extent must be a list, got {}".format(type(value))
            raise TypeError(msg)
        
        if self.profile is not None:
            p_aspect=self.shape[0]/self.shape[1]
            e_aspect=value[0]/value[1]
            if abs(e_aspect-p_aspect)>0.0001:
                msg="Extent aspect ratio doesn't match profile aspect ratio"
                raise ValueError(msg)
            else:
                self._extent=value
                self._grid_spacing=value[0]/self.shape[0]
        else:
            self.dimentions=len(value)
            if self.grid_spacing is not None:
                self.shape=[v/self.grid_spacing for v in value]
            
        return
    
    @extent.deleter
    def extent(self):
        self._extent=None
        self._grid_spacing=None
        if self.profile is None:
            del self.shape
    
    @property
    def shape(self):
        """
        The shape of the surface profile array, the number of points in each 
        direction
        """
        return self._shape
    
    @shape.setter
    def shape(self, value):
        if self.profile is None:
            self._shape=tuple([int(x) for x in value])
            self.size=np.prod(self._shape)
            if self.grid_spacing is not None:
                self._extent=[v*self.grid_spacing for v in value]
            elif self.extent is not None:
                e_aspect=self.extent[0]/self.extent[1]
                p_aspect=value[0]/value[1]
                if abs(e_aspect-p_aspect)<0.0001:
                    self._grid_spacing=(self.extent[0]/
                                        self.shape[0])
                else:
                    warnings.warn("Global size does not match profile size,"
                                  "global size has been deleted")
                    self._extent=None
        else:
            self._shape=self._profile.shape
            self.size=self._profile.size
        
    @shape.deleter
    def shape(self):
        if self.profile is None:
            self._shape=None
            self._size=None
        else:
            msg="Cannot delete shape with a surface profile set"
            raise AttributeError(msg)
    
    @property
    def profile(self):
        """
        The height data for the surface profile
        """
        return self._profile
    
    @profile.setter
    def profile(self, value):
        """Sets the profile property
        """
        try:
            self._profile=np.asarray(value, dtype=float)
        except ValueError:
            msg=("Could not convert profile to array of floats, profile contai"
                 "ns invalid values")
            raise ValueError(msg)
        
        self.shape=self._profile.shape
        self.dimentions=len(self._profile.shape)
        
        if self.grid_spacing is not None:
            self._extent=[self.grid_spacing*p for p in 
                               self.shape]
        elif self.extent is not None:
            if self.dimentions==1:
                self._grid_spacing=(self.extent[0]/ self.shape[0])
            if self.dimentions==2:
                e_aspect=self.extent[0]/self.extent[1]
                p_aspect=self.shape[0]/self.shape[1]
                
                if abs(e_aspect-p_aspect)<0.0001:
                    self._grid_spacing=(self.extent[0]/
                                        self.shape[0])
                else:
                    warnings.warn("Global size does not match profile size,"
                                  "global size has been deleted")
                    self._extent=None
    
    @profile.deleter
    def profile(self):
        self._profile=None
        del self.shape
        del self.extent
        del self.mask
        self.is_descrete=False
    
    @property
    def grid_spacing(self):
        return self._grid_spacing
    
    @grid_spacing.deleter
    def grid_spcaing(self):
        self._extent=None
        self._grid_spacing=None
        if self.profile is None:
            del self.shape
    
    
    @grid_spacing.setter
    def grid_spacing(self, grid_spacing):
        """ Change the grid spacing of the surface
        
        Changes the grid spacing attribute while keeping all other dimentions 
        up to date. Does not re interpolate on a different size grid, 
        stretches the surface to the new grid size keeping all points the same.
        
        """
        if grid_spacing is None:
            return
        
        if grid_spacing is not float:
            try:
                grid_spacing=float(grid_spacing)
            except ValueError:
                msg=("Invalid type, grid spacing of type {} could not be "
                     "converted into float".format(type(grid_spacing)))
                raise ValueError(msg)
        
        if np.isinf(grid_spacing):
            msg="Grid spacing must be finite"
            raise ValueError(msg)
        
        self._grid_spacing=grid_spacing
        
        if self.profile is None:
            if self.extent is not None:
                self._shape=[sz/grid_spacing for sz in 
                                         self.extent]
            elif self.shape is not None:
                self._extent=[grid_spacing*pt for pt in 
                                  self.shape]
        else:
            self._extent=[s*grid_spacing for s in self.shape]
        return
            
    def get_fft(self, profile_in=None):
        """ Find the fourier transform of the surface
        
        Findes the fft of the surface and stores it in your_instance.fft
        
        Parameters
        ----------
        profile_in : array-like optional (None)
            If set the fft of profile_in will be found and returned otherwise 
            instances profile attribue is used
        
        Returns
        -------
        transform : array
            The fft of the instance's profile or the profile_in if one is 
            supplied
            
        See Also
        --------
        get_psd
        get_acf
        show
        
        Notes
        -----
        Uses numpy fft.fft or fft.fft2 depending on the shape of the profile
        
        
        Examples
        --------
        >>># Set the fft property of the surface
        >>>my_surface.get_fft()
        
        >>># Return the fft of a provided profile
        >>>fft_of_profile_2=my_surface.get_fft(profile_2)
        
        """
        if profile_in is None:
            profile=self.profile
        else:
            profile=profile_in
        try:
            if len(profile.shape)==1:
                transform=np.fft.fft(profile)
                if type(profile_in) is bool: self.fft=transform
            else:
                transform=np.fft.fft2(profile)
                if type(profile_in) is bool: self.fft=transform
        except AttributeError:
            raise AttributeError('Surface must have a defined profile for fft'
                                 ' to be used')
        
        return transform
    
    def get_acf(self, profile_in=None):
        """ Find the auto corelation function of the surface
        
        Findes the ACF of the surface and stores it in your_instance.acf
        
        Parameters
        ----------
        profile_in : array-like optional (None)
        
        Returns
        -------
        output : ACF object
            An acf object with the acf data stored, the values can be extracted
            by numpy.array(output)
            
        See Also
        --------
        get_psd
        get_fft
        show
        slippy.surface.ACF
        
        Notes
        -----
        ACF data is kept in ACF objects, these can then be interpolated or 
        evaluated at specific points with a call:
        
        
        Examples
        --------
        
        >>> # Sets the acf property of the surface with an ACF object
        >>> my_surface.get_acf()
        >>> # The acf values are then given by the following
        >>> numpy.array(my_surface.acf)
        >>> # The acf can be shown using the show function:
        >>> my_surface.show('acf', 'image')
        
        >>> # Finding the ACF of a provided profile:
        >>> ACF_object_for_profile_2=my_surface.get_acf(profile_2)
        >>> # equvalent to ACF(profile_2)
        
        """
        
        if profile_in is None:
            self.acf=ACF(self)
        else:
            profile=np.asarray(profile_in)
            x=profile.shape[0]
            y=profile.shape[1]
            output=(scipy.signal.correlate(profile,profile,'same')/(x*y))
            return output

    def get_psd(self): 
        """ Find the power spectral density of the surface
        
        Findes the PSD of the surface and stores it in your_instance.psd
        
        Parameters
        ----------
        
        (None)
        
        Returns
        -------
        
        (None), sets the psd attribute of the instance
            
        See Also
        --------
        get_fft
        get_acf
        show
        
        Notes
        -----
        Finds the psd by fourier transforming the ACF, in doing so looks for 
        the instance's acf property. If this is not found the acf is calculated
        and set.
        
        Examples
        --------
        
        >>> # sets the psd attribute of my_surface
        >>> my_surface.get_psd()
        
        """
        # PSD is the fft of the ACF (https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density)
        if self.acf is None:
            self.get_acf()
        self.psd=self.get_fft(np.asarray(self.acf))
    
    def subtract_polynomial(self, order, mask=None):
        """ Flatten the surface by subtracting a polynomial 
        
        Alias for :func:`~slippy.surface.subtract_polynomial` function
        
        
        """
        if mask is None:
            mask=self.mask
        
        new_profile, coefs = subtract_polynomial(self.profile,order,mask)
        self.profile=new_profile
        
        return coefs
    
    __all__=['roughness', 'subtract_polynomial', 'get_mat_vr',
         'get_height_of_mat_vr', 'get_summit_curvatures',
         'find_summits', 'low_pass_filter']
    
    def roughness(self, parameter_name, mask=None, curved_surface=False, 
                  no_flattening=False, filter_cut_off=False, 
                  four_nearest=False):
        """ Find areal roughness parameters
        
        Alias for :func:`~slippy.surface.roughness` function
        
        """
        if mask is None:
            mask=self.mask
        
        out=roughness(self, parameter_name, mask=mask, 
                      curved_surface=curved_surface, 
                      no_flattening=no_flattening, 
                      filter_cut_off=filter_cut_off, 
                      four_nearest=four_nearest)
        return out
    
    def get_mat_vr(self, height, void=False, mask=None, ratio=True):
        """ Find the material or void volume ratio for a given height
        
        Alias for :func:`~slippy.surface.get_mat_vr` function
        
        """
        if mask is None:
            mask=self.mask
        
        return get_mat_vr(height, self, void=void, mask=mask,
                                 ratio=ratio)
    
    def get_height_of_mat_vr(self, ratio, void=False, mask=None, 
                             accuracy=0.001):
        """ Find the height of a given material or void volume ratio
        
        Alias for :func:`~slippy.surface.get_height_of_mat_vr` function
        
        """
        if mask is None:
            mask=self.mask
            
        return get_height_of_mat_vr(ratio, self, void=void, mask=mask, 
                                       accuracy=accuracy)

    def get_summit_curvature(self, summits=None, mask=None, 
                             filter_cut_off=None, four_nearest=False):
        """ Get summit curvatures
        
        Alias for :func:`~slippy.surface.get_summit_curvature` function
        
        """
        if mask is None:
            mask=self.mask
        
        return get_summit_curvatures(self, summits=summits, mask=mask,
                                     filter_cut_off=filter_cut_off, 
                                     four_nearest=four_nearest)
    
    def find_summits(self, mask=None, four_nearest=False, filter_cut_off=None,
                     invert=False):
        """ Find summits after low pass filtering
        
        Alias for :func:`~slippy.surface.find_summits` function
        
        """
        if mask is None:
            mask=self.mask
        
        if invert:
            return find_summits(self.profile*-1, 
                                grid_spacing=self.grid_spacing, mask=mask, 
                                four_nearest=four_nearest, 
                                filter_cut_off=filter_cut_off)
        else:
            return find_summits(self, mask=mask, four_nearest=four_nearest, 
                                filter_cut_off=filter_cut_off)
        
    def low_pass_filter(self, cut_off_freq, return_copy=False):
        """ Low pass FIR filter the surface profile
        
        Alias for :func:`~slippy.surface.low_pass_filter` function
        
        """
        if return_copy:
            return low_pass_filter(self, cut_off_freq)
        else:
            self.profile=low_pass_filter(self, cut_off_freq)
    
    def read_from_file(self, path, **kwargs):
        """ Read profile data from a file
        
        Reads depth data from files, supported files are: .txt, .csv, .al3d and
        .mat
        
        Parameters
        ----------
        path : str
            The path to a data file including the file extention
        
        Other Parameters
        ----------------
        
        delim : str optional (',')
            The delimiter used in the data file, only needed for csv or txt
            files
        p_name : str optional ('profile')
            The name of the variable containing the profile data, only needed 
            for mat files
        gs_name : str optional ('grid_spacing')
            The name of the variable containing the grid_spacing, only needed 
            for mat files
        
        Returns
        -------
        
        kwargs - dict
            The keyword arguments with the used keys removed
        
        See Also
        --------
        read_surface
        alicona_read
        
        Notes
        -----
        This method is invoked by the initialisation method if the keyword 
        file_name is passed to the constructor. It can also be invoked more 
        simply by calling the read_surface function
        
        Examples
        --------
        
        >>> # read an alicona file in
        >>> import slippy.surface as S
        >>> my_surface = S.Surface()
        >>> my_surface.read_from_file('data.al3d')
        
        """
        
        file_ext=os.path.splitext(path)[1]
        
        try:
            file_ext=file_ext.lower()
        except AttributeError:
            msg="file_type should be a string"
            raise ValueError(msg)
        if file_ext=='.csv' or file_ext=='.txt':
            import csv
            if 'delim' in kwargs:
                delimiter=kwargs.pop('delim')
            elif 'delimiter' in kwargs:
                delimiter=kwargs.pop('delimiter')
            else:
                delimiter=','
            with open(path) as file:
#                if delimiter == ' ' or delimiter == '\t':
#                    reader=csv.reader(file, delim_whitespace=True)
#                else:
                reader=csv.reader(file, delimiter=delimiter)
                profile=[]
                for row in reader:
                    if row:
                        if type(row[0]) is float:
                            profile.append(row)
                        else:
                            if len(row)==1:
                                try:
                                    row=[float(x) for x in row[0].split() 
                                         if not x=='']
                                    profile.append(row)
                                except ValueError:
                                    pass
                            else:
                                try:
                                    row=[float(x) for x in row if not x=='']
                                    profile.append(row)
                                except ValueError:
                                    pass
                        
            self.profile=np.array(profile)
            
        elif file_ext=='.al3d':
            from .alicona import alicona_read
            data=alicona_read(path)
            self.profile=data['DepthData']
            self.grid_spacing=data['Header']['PixelSizeXMeter']
            
        elif file_ext=='.mat':
            kwargs=self._read_mat(path, kwargs)
            
        else:
            msg=('Path does not have a recognised file extention')
            raise ValueError(msg)
        self.is_descrete=True
        return kwargs
    
    def _read_mat(self, path, kwargs):
        """ Reads .mat files as surfaces 
        
        Parameters
        ----------
        path : str
            full path including file name to a .mat file
        kwargs : dict
            with fields of p_name and gs_name giving the name of the variables
            for the profile data and grid spacing respectively
        
        Retruns
        -------
        kwargs : dict
            The kwargs dict with the p_name and gs_name items removed
            
        Notes
        -----
        This method will search the .mat file for the given keys. If no keys 
        are given, and the .mat file contains variables called grid_spacing or
        profile these are set as the relavent attributes. Otherwise, if the 
        .mat file only contains one variable this is set as the profile. 
        If none of the above are true, or if the given keys are not found
        an error is raised
            
        """
        # load file
        mat = loadmat(path)
        keys=[key for key in mat.keys if not key.startswith('_')]
        
        #try to find keys
        key_found=False
        
        if 'gs_name' in kwargs:
            gs_name=kwargs.pop('gs_name')
            try:
                self.grid_spcaing=mat[gs_name]
                key_found=True
            except KeyError:
                msg=("Name {} not found in .mat file,".format(gs_name)+
                     " names found were: ".join(keys))
                raise ValueError(msg)
        elif 'grid_spacing' in keys:
            self.grid_spacing=mat['grid_spacing']
            key_found=True
        
        if 'p_name' in kwargs:
            p_name=kwargs.pop('p_name')
            try:
                self.profile=mat[p_name]
                key_found=True
            except KeyError:
                msg=("Name {} not found in .mat file,".format(p_name)+
                     " names found were: ".join(keys))
                raise ValueError(msg)    
        elif 'profile' in keys:
            self.profile=mat['profile']
            key_found=True
        
        # if keys not found check if there is only one item in mat file
        if len(keys)==1 and not key_found:
            self.profile=mat(keys[0])
        elif not key_found:
            msg=("Profile not found, the p_name and gs_name key words must be "
                 "set to specify the names of the variables which contain the "
                 "profile data and grid spacing names found in this .mat file "
                 "are: ".join(keys))
            raise ValueError(msg)
        
        return kwargs
    
    def resample(self, new_grid_spacing, return_profile=False, 
                 remake_interpolator=False):
        """ Resample the profile by interpolation
        
        Changes the grid spacing of a surface by intrpolation of the original
        surface profile
        
        Parameters
        ----------
        
        new_grid_spacing : float
            The new grid spacing to be interpolated to
        return_profile : bool optional (False)
            If true the interpolated profile is returned otherwise it is set as
            the profile attribute of the instance
        remake_interpolator : boo optional (False)
            If true the interpolator will be remade before interpolation
            see notes
            
        Returns
        -------
        
        inter_profile : array
            If return_profile is true the interpolated profile is returned
            
        See Also
        --------
        rotate
        fill_holes
        surface_like
        
        Notes
        -----
        On the first call this function will make an interpolator object which 
        is used to interpolate, on subsequent calls this object is found and 
        used resulting in no loss of quality. If the remake_interpolator key
        word is set to true this interpolator is remade. This will result in a
        loss of quality for subsequent calls but is necessary if the profile 
        property has changed.
        
        This method does not support masking. 
        
        The profile should have nan or inf values removed by the fill_holes 
        method before running this
        
        Examples
        --------
        
        >>> import numpy as np 
        >>> import slippy.surface as S
        >>> profile=np.random.normal(size=(101,101))
        >>> my_surface=S.assurface(profile, grid_spacing=1)
        >>> # interpolate on a corse grid:
        >>> my_surface.resample(10)
        >>> # check shape:
        >>> my_surface.shape
        (11,11)
        >>> # restore original profile:
        >>> my_surface.resample(1)
        >>> my_surface.shape
        (101,101)
        """
        if remake_interpolator or not self._inter_func:
            import scipy.interpolate
            x0=np.arange(0, self.extent[0], self.grid_spacing)
            y0=np.arange(0, self.extent[1], self.grid_spacing)
            self._inter_func=scipy.interpolate.RectBivariateSpline(x0, y0, 
                                                                  self.profile)
        x1=np.arange(0, self.extent[0], new_grid_spacing)
        y1=np.arange(0, self.extent[1], new_grid_spacing)
        inter_profile=self._inter_func(x1,y1)
        
        if return_profile:
            return inter_profile
        else:
            self.profile=inter_profile
            self.grid_spacing=new_grid_spacing
        
    def fill_holes(self, hole_value='auto', mk_copy=False, remove_boarder=True, 
                   b_thresh=0.99):
        """ Replaces specificed values with filler
        
        Removes boarder then uses biharmonic equations algorithm to fill holes 
        
        Parameters
        ----------
        hole_value: {'auto' or float}
            The value to be replaced, 'auto' replaces all -inf, inf and nan 
            values
        mk_copy : bool
            if set to true a new surface object will be returned with the holes 
            filled otherwise the profile property of the current surface is 
            updated
        remove_boarder : bool
            Defaults to true, removes the boarder from the image until the 
            first row and column that have 
        
        Returns
        -------
        If mk_copy is true a new surface object with holes filled else resets 
        profile property of the instance and returns nothing
        
        See Also
        --------
        skimage.restoration.inpaint.inpaint_biharmonic
        
        Notes
        -----
        When alicona images are imported the invalid pixel value is 
        automatically set to nan so this will work in auto mode
        
        Holes are filled with bi harmonic equations
        
        Examples
        --------
        
        >>> # make a dummy profile
        >>> x=np.arange(12, dtype=float)
        >>> X,_=np.meshgrid(x,x)
        >>> # pad with nan values
        >>> X2=np.pad(X,2,'constant', constant_values=float('nan'))
        >>> # add hole to centre
        >>> X2[6,6]=float('nan')
        >>> # make surface
        >>> my_surface=S.Surface(profile=X2)
        >>> my_surface.fill_holes()
        >>> my_surface.profile[6,6]
        6.0
        """
        from skimage.restoration import inpaint
        
        profile=self.profile
        
        if hole_value=='auto':
            holes=np.logical_or(np.isnan(profile), np.isinf(profile))
        else:
            holes=profile==hole_value
        if sum(sum(holes))==0:
            warnings.warn('No holes detected')

        profile[holes]=0

        if remove_boarder:
            # find rows
            good=[False]*4
            
            start_r=0
            end_r=len(profile)-1
            start_c=0
            end_c=len(profile[0])-1
            
            while not(all(good)):
                #start row
                if 1-sum(holes[start_r,start_c:end_c])/(end_c-start_c)<b_thresh:
                    start_r+=1
                else:
                    good[0]=True
                
                #end row
                if 1-sum(holes[end_r,start_c:end_c])/(end_c-start_c)<b_thresh:
                    end_r-=1
                else:
                    good[1]=True
            
                if 1-sum(holes[start_r:end_r,start_c])/(end_r-start_r)<b_thresh:
                    start_c+=1
                else:
                    good[2]=True
                
                if 1-sum(holes[start_r:end_r,end_c])/(end_r-start_r)<b_thresh:
                    end_c-=1
                else:
                    good[3]=True
           
            profile=profile[start_r:end_r, start_c:end_c]
            holes=holes[start_r:end_r, start_c:end_c]

        profile_out = inpaint.inpaint_biharmonic(profile, holes,
                multichannel=False)
        
        if mk_copy:
            new_surf=Surface(profile=profile_out, 
                             grid_spacing=self.grid_spacing,
                             is_descrete=True)
            return new_surf
        else:
            self.profile=profile_out
            return
        
        
    def __add__(self, other):
        if self.extent==other.extent and self.extent is not None:
            if self.grid_spacing==other.grid_spacing:
                out=Surface(profile=self.profile+other.profile, 
                            grid_spacing=self.grid_spacing, 
                            is_descrete=self.is_descrete)
                return out
            else:
                msg="Surface sizes do not match: resampling"
                warnings.warn(msg)
                ## resample surface with coarser grid then add again
                if self.grid_spacing>other.grid_spacing:
                    self.resample(other.grid_spacing, False)
                else:
                    other.resample(self.grid_spacing)
                    
                return self+other
            
        elif self.shape==other.shape:
            if self.grid_spacing is not None:
                if other.grid_spacing is not None:
                    if self.grid_spacing==other.grid_spacing:
                        gs=self.grid_spacing
                    else:    
                        msg=('Surfaces have diferent sizes and grid spacing is' 
                             ' set for both for element wise adding delete the'
                             ' grid spacing for one of the surfaces using: '
                             'del surface.grid_spacing before adding')
                        raise AttributeError(msg)
                else: #only self not other
                    gs=self.grid_spacing
            else: #not self
                gs=other.grid_spacing
                
            out=Surface(profile=self.profile+other.profile,
                            grid_spacing=gs, 
                            is_descrete=self.is_descrete)
            return out
        else:
            raise ValueError('surfaces are not compatible sizes cannot add')
            
    def __sub__(self, other):
        if self.extent==other._extent and self.extent:
            if self.grid_spacing==other.grid_spacing:
                out=Surface(profile=self.profile-other.profile, 
                            grid_spacing=self.grid_spacing, 
                            is_descrete=self.is_descrete)
                return out
            else:
                msg="Surface sizes do not match: resampling"
                warnings.warn(msg)
                ## resample surface with coarser grid then add again
                if self.grid_spacing>other.grid_spacing:
                    self.resample(other.grid_spacing, False)
                else:
                    other.resample(self.grid_spacing)
                    
                return self-other
            
        elif self.shape==other.shape:
            if self.grid_spacing:
                if other.grid_spacing:
                    if self.grid_spacing==other.grid_spacing:
                        gs=self.grid_spacing
                    else:    
                        msg=('Surfaces have diferent sizes and grid spacing is' 
                             ' set for both for element wise adding unset the '
                             'grid spacing for one of the surfaces using: '
                             'set_grid_spacing(None) before subtracting')
                        raise AttributeError(msg)
                else: #only self not other
                    gs=self.grid_spacing
            else: #not self
                gs=other.grid_spacing
                
            out=Surface(profile=self.profile-other.profile,
                            grid_spacing=gs, 
                            is_descrete=True)
            return out
        else:
            raise ValueError('surfaces are not compatible sizes cannot'
                             ' subtract')
        
    def show(self, property_to_plot='profile', plot_type='default', ax=False,
             dist=None, stride=None):
        """ Polt surface properties
        
        Parameters
        ----------
        property_to_plot : str or list of str length N optional ('profile')
            The property to be plotted see notes for supported names
        plot_type : str or list of str length N optional ('defalut')
            The type of plot to be produced, see notes for supported types
        ax : matplotlib axes or False optional (False)
            If supploed the plot will be added to the axis
        
        Returns
        -------
        ax : matplotlib axes or list of matplotlib axes length N
            The axis with the plot
            
        Other Parameters
        ----------------
        
        dist : a scipy probability distribution optional (None)
            Only used if probplot is requested, the probability distribution
            to plot against
        stride : float optional (None)
            Only used if a wire frame plot is requested, the stride between 
            wires
        
        See Also
        --------
        get_fft
        get_psd
        get_acf
        ACF
        
        Notes
        -----
        If fft, psd or acf are requested the field of the surface is filled 
        by the relavent get_ method before plotting. 
        
        The grid spacing attribute should be set before plotting
        
        2D and 1D plots can be produced. 2D properties are:
            
            - profile - surface profile
            - fft2d   - fft of the surface profile
            - psd     - power spectral density of the surface profile
            - acf     - auto correlation function of the surface
            - apsd    - angular powerspectral density of the profile
        
        Plot types allowed for 2D plots are:
            
            - surface (default)
            - image
            - mesh
            
        If a mesh plot is requested the distance between lines in the mesh can
        be specified with the stride keyword
            
        1D properties are:
            
            - histogram - histogram of the profile heights
            - fft1d     - 1 dimentional fft of the surface
            - qq        - quantile quantile plot of the surface heights
            - disthist  - a histogram with a distribution drawn over
        
        If qq or dist hist are requested the distribution to be plotted against
        the height values can be specified by the dist keyword
        
        Each of the 1D properties can only be plotted on it's default plot type
        
        Examples
        --------
        >>> # show the surface profile as an image:
        >>> my_surface.show('profile', 'image')
        
        >>> # show the 2D fft of the surface profile with a range of plot types
        >>> self.show(['fft2D','fft2D','fft2D'], ['mesh', 'image', 'default'])
           
        """
        
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        
        types2d=['profile', 'fft2d', 'psd', 'acf', 'apsd']
        types1d=['histogram','fft1d', 'qq', 'disthist']
        
        # using a recursive call to deal with multiple plots on the same fig
        if type(property_to_plot) is list:
            number_of_subplots=len(property_to_plot)
            if not type(ax) is bool:
                msg=("Can't plot multiple plots on single axis, "
                     'making new figure')
                warnings.warn(msg)
            if type(plot_type) is list:
                if len(plot_type)<number_of_subplots:
                    plot_type.extend(['default']*(number_of_subplots-
                                                  len(plot_type)))
            else:
                plot_type=[plot_type]*number_of_subplots
            # 11, 12, 13, 22, then filling up rows of 3 (unlikely to be used)
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')
            if len(property_to_plot)<5:
                n_cols=[1,2,3,2][number_of_subplots-1]
            else:
                n_cols=3
            n_rows=int(ceil(number_of_subplots/3))
            fig = plt.figure()
            ax=[]
            sub_plot_number=100*n_rows+10*n_cols+1
            print(sub_plot_number)
            for i in range(number_of_subplots):
                if property_to_plot[i].lower() in types2d and not plot_type[i]=='image':
                    ax.append(fig.add_subplot(sub_plot_number+i, projection='3d'))
                else:
                    ax.append(fig.add_subplot(sub_plot_number+i))
                self.show(property_to_plot[i], plot_type[i], ax[i])
            fig.show()
            return fig, ax
        #######################################################################
        ####################### main method ###################################
        # 2D plots
        try:
            property_to_plot=property_to_plot.lower()
        except AttributeError:
            msg="Property to plot must be a string or a list of strings"
            raise ValueError(msg)
        
        if not (property_to_plot in types2d or property_to_plot in types1d):
            msg=('Unsupported property to plot see documentation for details'
                 ', type given: \n' + str(property_to_plot) + ' \nsupported ty'
                 'pes: \n' + ' '.join(types2d+types1d))
            raise ValueError(msg)
            return
        
        if not ax:
            fig = plt.figure()
        
        if property_to_plot in types2d:
            if not ax and not plot_type=='image':
                ax=fig.add_subplot(111, projection='3d')
            elif not ax and plot_type=='image':
                ax=fig.add_subplot(111)
                
            if property_to_plot=='profile':
                labels=['Surface profile', 'x', 'y', 'Height']
                Z=self.profile
                x=self.grid_spacing*np.arange(self.shape[0])
                y=self.grid_spacing*np.arange(self.shape[1])
                
            elif property_to_plot=='fft2d':
                labels=['Fourier transform of surface', 'u', 'v', '|F(x)|']
                if self.fft is None:
                    self.get_fft()
                Z=np.abs(np.fft.fftshift(self.fft))
                x=np.fft.fftfreq(self.shape[0], self.grid_spacing)
                y=np.fft.fftfreq(self.shape[1], self.grid_spacing)
                
            elif property_to_plot=='psd':
                labels=['Power spectral density', 'u', 'v', 'Power/ frequency']
                if self.psd is None:
                    self.get_psd()
                Z=np.abs(np.fft.fftshift(self.psd))
                x=np.fft.fftfreq(self.shape[0], self.grid_spacing)
                y=np.fft.fftfreq(self.shape[1], self.grid_spacing)
                
            elif property_to_plot=='acf':
                labels=['Auto corelation function', 'x', 'y', 
                        'Surface auto correlation']
                if self.acf is None:
                    self.get_acf()
                Z=np.abs(np.asarray(self.acf))
                x=self.grid_spacing*np.arange(self.shape[0])
                y=self.grid_spacing*np.arange(self.shape[1])
                x=x-max(x)/2
                y=y-max(y)/2
            elif property_to_plot=='apsd':
                labels=['Angular power spectral density', 'x', 'y']
                if self.fft is None:
                    self.get_fft()
                p_area=(self.shape[0]-1)*(self.shape[1]-1)*self.grid_spacing**2
                Z=self.fft*np.conj(self.fft)/p_area
                x=self.grid_spacing*np.arange(self.shape[0])
                y=self.grid_spacing*np.arange(self.shape[1])
                x=x-max(x)/2
                y=y-max(y)/2 
            
            X,Y=np.meshgrid(x,y)
            print(X.shape)
            
            if plot_type=='default' or plot_type=='surface':
                ax.plot_surface(X,Y,np.transpose(Z))
                plt.axis('equal')
                ax.set_zlabel(labels[3])
            elif plot_type=='mesh':
                if property_to_plot=='psd' or property_to_plot=='fft2d':
                    X, Y = np.fft.fftshift(X), np.fft.fftshift(Y)
                if stride:
                    ax.plot_wireframe(X, Y, np.transpose(Z), rstride=stride, 
                                      cstride=stride)
                else:
                    ax.plot_wireframe(X, Y, np.transpose(Z), rstride=25, 
                                      cstride=25)
                ax.set_zlabel(labels[3])
            elif plot_type=='image':
                ax.imshow(Z, extent=[min(y),max(y),min(x),max(x)], aspect=1)
            else:
                ValueError('Unrecognised plot type')
            
            ax.set_title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
            return ax
        
        #######################################################################
        ############## 1D plots ###############################################
        #######################################################################
        
        elif property_to_plot in types1d:
            if not ax:
                ax= fig.add_subplot(111)
                
            if property_to_plot=='histogram':
                # do all plotting in this loop for 1D plots
                labels=['Histogram of sufrface heights', 'height', 'counts']
                ax.hist(self.profile.flatten(), 100)
                
            elif property_to_plot=='fft1d':
                if self.dimentions==1:
                    labels=['FFt of surface', 'frequency', '|F(x)|']
                    
                    if type(self.fft) is bool:
                        self.get_fft()
                    x=np.fft.fftfreq(self.shape[0], self.grid_spacing)
                    y=np.abs(self.fft/self.shape[0])
                    # line plot for 1d surfaces
                    ax.plot(x,y)
                    ax.xlim(0,max(x))
                else:
                    labels=['Scatter of frequency magnitudes', 
                            'frequency', '|F(x)|']
                    u=np.fft.fftfreq(self.shape[0], self.grid_spacing)
                    v=np.fft.fftfreq(self.shape[1], self.grid_spacing)
                    U,V=np.meshgrid(u,v)
                    freqs=U+V
                    if type(self.fft) is bool:
                        self.get_fft()
                    mags=np.abs(self.fft)
                    # scatter plot for 2d frequencies
                    ax.scatter(freqs.flatten(), mags.flatten(), 0.5, None, 'x')
                    ax.set_xlim(0,max(freqs.flatten()))
                    ax.set_ylim(0,max(mags.flatten()))
            elif property_to_plot=='disthist':
                import seaborn
                labels=['Histogram of sufrface heights', 'height', 'P(height)']
                if dist:
                    seaborn.distplot(self.profile.flatten(), fit=dist, 
                                 kde=False, ax=ax)
                else:
                    seaborn.distplot(self.profile.flatten(), ax=ax)    
            elif property_to_plot=='qq':
                from scipy.stats import probplot
                labels=['Probability plot', 'Theoretical quantities', 
                        'Ordered values']
                if dist:
                    probplot(self.profile.flatten(), dist=dist, fit=True,
                             plot=ax)
                else:
                    probplot(self.profile.flatten(), fit=True, plot=ax)
            ax.set_title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
            return ax
        #######################################################################
        #######################################################################
            
    def __array__(self):
        return np.asarray(self._profile)

    def _get_points_from_extent(self, extent, grid_spacing):
        if type(grid_spacing) in [int, float, np.float32, np.float64]:
            grid_spacing=[grid_spacing]*2
        if type(extent[0]) in [int, float, np.float32, np.float64]:
            extent=[[0,extent[0]],[0,extent[1]]]
        x=np.arange(extent[0][0], extent[0][1], grid_spacing[0])
        y=np.arange(extent[1][0], extent[1][1], grid_spacing[1])
        
        X,Y=np.meshgrid(x,y)
        
        return(X,Y)
    
    def rotate(self, radians):
        """Rotate the surface relative to the grid and reinterpolate
        """
        raise NotImplementedError('Not implemented yet')
    
    def _descretise_checks(self):
        if self.is_descrete:
            msg=('Surface is already discrete this will overwrite surface'
                 ' profile')
            raise warnings.warn(msg)
        if self.grid_spacing is None:
            msg='A grid spacing must be provided before descretisation'
            raise AttributeError(msg)
        else:
            grid_spacing=self.grid_spacing
            
        try:
            shape=[int(ex/grid_spacing) for ex in self.extent]
            size=1
            for pts in shape:
                size *= pts
            self.size=size
            self.shape=shape
        except TypeError:
            msg='Extent must be set before descretisation'
            raise AttributeError(msg)
        if size>10E7:
            warnings.warn('surface contains over 10^7 points calculations will'
                          ' be slow, consider splitting surface for analysis')
        return    

if __name__=='__main__':
    A=Surface(file_name='C:\\Users\\44779\\code\\SlipPY\\data\\image1_no head'
              'er_units in nm.txt', delimiter=' ', grid_spacing=0.001)
    #testing show()
    #types2d=['profile', 'fft2d', 'psd', 'acf', 'apsd']
    #types1d=['histogram','fft1d', 'qq', 'disthist']
    #A.show(['profile','fft2d','acf'], ['surface', 'mesh', 'image'])
    #A.show(['histogram', 'fft1d', 'qq', 'disthist'])
    #out=A.birmingham(['sa', 'sq', 'ssk', 'sku'])
    #out=A.birmingham(['sds','sz', 'ssc'])
    #out=A.birmingham(['std','sbi', 'svi', 'str'])
    #out=A.birmingham(['sdr', 'sci', 'str'])
    