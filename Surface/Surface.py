import os
import numpy as np
import warnings
from math import ceil
import scipy.signal

__all__=['Surface']

class Surface(object):
    """ Object for reading, manipulating and plotting surfaces
    
    The basic surface class contains methods for setting properties, 
    examining measures of roughness and descriptions of surfaces, plotting,
    fixing and editing surfaces.
    
    Parameters
    ----------
    
    file_name : str optional (None)
        The file name including the extention of file, for vaild types see 
        notes
    delim : str optional (',')
        The delimitor used in the file, only needed for csv or txt files
    profile : array-like optional (np.array([]))
        A surface profile 
    global_size : 2 element list optional (None)
        The size of the surface in each dimention
    grid_spacing : float optional (None)
        The distance between nodes on the grid, 
    dimentions : int optional (2)
        The number of diimentions of the surface
    
    Attributes
    ----------
    acf, psd, fft : array or None
        The acf, psd and fft of the surface set by the get_acf get_psd and
        get_fft methods
    
    profile : array
        The height infromation of the surface
    
    surface_type : str
        A description of the surface type
    
    Methods
    -------
    
    set_global_size
    set_grid_spacing
    get_fft
    get_acf
    get_psd
    
    
    See Also
    --------
    
    
    Notes
    -----
    
    
    Examples
    --------
    
    
    
    
    #TODO:
        priority
        get_quantiles(n) # n is the number of quantiles needed... might not be necessary or might want to change how hist (in show) works to work with it
        rotate 
        
        Then start checking stuff
        make init work with these options 
        Add desciption and example of each method 
        check summit finding
        
        def mesh
        
        
        Make this pretty ^
    """
    # The surface class for descrete surfaces (typically experiemntal)
    is_descrete=False
    acf=None
    aacf=None
    psd=None
    fft=None
    hist=None
    profile=np.array([])
    sa=None
    surface_type="Generic"
    dimentions=2
    _grid_spacing=None
    _pts_each_direction=[]
    _global_size=[]
    _inter_func=None
    _allowed_keys=[]
    
    def __init__(self,**kwargs):
        # initialisation surface
        kwargs=self._init_checks(kwargs)
        # check for file
        if 'file_name' in kwargs:
            ext=os.path.splitext(kwargs['file_name'])
            file_name=kwargs.pop('file_name')
            if ext=='pkl':
                self.load_from_file(file_name)
            else:
                kwargs=self.read_from_file(file_name, **kwargs)
        
        # at this point everyone should have taken what they needed
        if kwargs:
            print(kwargs)
            raise ValueError("Unrecognised keys in keywords")
        
    def _init_checks(self, kwargs={}):
        # add anything you want to run for all surface types here
        if kwargs:
            allowed_keys=self._allowed_keys
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
            
            for key in allowed_keys:
                if key in kwargs:
                    del kwargs[key]
            
            if 'profile' in kwargs:
                self.profile=np.asarray(kwargs.pop['profile'])
                self.is_descrete=True
                p_set=True
            else:
                p_set=False
            
            if 'pts_each_direction' in kwargs:
                if p_set or 'file_name' in kwargs:
                    msg=('Cannot set the surface profile and the number of '
                         'points independently')
                    raise ValueError(msg)
                self.pts_each_direction=kwargs.pop('pts_each_direction')
                pts_set=True
            else:
                pts_set=False   
            
            if 'grid_spacing' in kwargs:
                self.set_grid_spacing(kwargs.pop('grid_spacing'))
                gs_set=True
            else:
                gs_set=False
            
            if 'global_size' in kwargs:
                if pts_set and gs_set:
                    msg=('Too many dimentions provided, unset one of :'
                        'grid_size, pts_each_direction, or global_size')
                    raise ValueError(msg)
                if gs_set and p_set:
                    msg=('Only grid_spacing or global size can be set with a '
                        'specific profile, it is recomended to only set grid_'
                        'spacing')
                    raise ValueError(msg)
                if p_set:
                    msg=('global_size set with profile, only the first '
                        'dimention of the surface will be used to calculate '
                        'the grid spacing')
                    warnings.warn(msg)
                self.set_global_size(kwargs.pop('global_size'))
            
        return kwargs
    
    def set_global_size(self, global_size):
        """ Changes the global size of the surface
        
        Sets the global size of the surface without reinterpolation, keeps all
        other dimnetions up to date
        
        Parameters
        ----------
        global_size : 2 element list
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
        self._global_size=global_size
        
        if global_size is None:
            return
        
        if self.profile.size==0:
            if self._grid_spacing:
                self._pts_each_direction=[sz/self._grid_spacing for sz in 
                                         global_size]
            elif self._pts_each_direction:
                gs1=global_size[0]/self._pts_each_direction[0]
                gs2=global_size[1]/self._pts_each_direction[1]
                if abs(gs1-gs2)/gs1>0.001:
                    msg=('Incompatable global size and points in each '
                         'direction would result in non square grid. '
                         'grid spacing not set, points in each direction '
                         'deleted')
                    warnings.warn(msg)
                    self._pts_each_direction=[]
                else:
                    self._grid_spacing=(global_size[0]/
                                        self._pts_each_direction[0])
        else:
            gs1=global_size[0]/self.profile.shape[0]
            gs2=global_size[1]/self.profile.shape[1]
            if abs(gs1-gs2)/gs1>0.001:
                shape=self.profile.shape
                msg=('Incompatible global size and surface size, would result'
                     ' in non square grid. global size was {global_size}, '
                     'profile shape is {shape}'.format(**locals()))
                raise ValueError(msg)    
            else:
                self._grid_spacing=global_size[0]/self.profile.shape[0]
                self._pts_each_direction=self.profile.shape
        
        return
        
    
    def set_grid_spacing(self, grid_spacing):
        """ Change the grid spacing of the surface
        
        Changes the grid spacing attribute while keeping all other dimentions 
        up to date. Does not re interpolate on a different size grid, 
        stretches the surface to the new grid size keeping all points the same.
        
        Parameters
        ----------
        grid_spacing : float
            The grid spacing to be set in length units
        
        Returns
        -------
        
        Nothing
        
        See Also
        --------
        resample - changes the grid spacing with reinterpolation
        set_global_size
        
        Notes
        -----
        
        Use this to set dimentions of the surface rather than setting directly 
        should keep all the other dimentions up to date.
        
        Examples
        --------
        
        
        """
        self._grid_spacing=grid_spacing
        
        if grid_spacing is None:
            return
        
        if self.profile.size==0:
            if self._global_size:
                self._pts_each_direction=[sz/grid_spacing for sz in 
                                         self._global_size]
            elif self._pts_each_direction:
                self._global_size=[grid_spacing*pt for pt in 
                                  self._pts_each_direction]
        else:
            self._global_size=[sp*grid_spacing for sp in self.profile.shape]
            self._pts_each_direction=self.profile.shape
        
        return
            
    def get_fft(self, profile_in=None):
        """ Find the fourier transform of the surface
        
        Findes the fft of the surface and stores it in your_instance.fft
        
        Parameters
        ----------
        surf_in : array-like optional (None)
        
        Returns
        -------
        transform : array
            The fft of the instance's profile or the profile_in if one is 
            supplied
            
        Examples
        --------
        >>># Set the fft property of the surface
        >>>my_surface.get_fft()
        
        >>># Return the fft of a provided profile
        >>>fft_of_profile_2=my_surface.get_fft(profile_2)
        
        See Also
        --------
        get_psd
        get_acf
        show
        
        Notes
        -----
        Uses numpy fft.fft or fft.fft2 depending on the shape of the profile
        
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
            
        Examples
        --------
        
        >>>my_surface.get_acf()
        Sets the acf property of the surface with an ACF object
        
        >>>numpy.array(my_surface.acf)
        Then gives the acf values
        
        >>>ACF_object_for_profile_2=my_surface.get_acf(profile_2)
        Returns the ACF of a provided profile, equvalent to ACF(profile_2)
        
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
        >>>acf_data_grid=my_acf_object(new_x_pts,new_y_pts)
        
        """
        
        if profile_in is None:
            ACF=__import__(__name__).surface.ACF
            self.acf=ACF(self)
        else:
            profile=np.asarray(profile_in)
            x=profile.shape[0]
            y=profile.shape[1]
            output=(scipy.signal.correlate(profile,profile,'same')/(x*y))
            return output

    def get_psd(self): 
        """ Find the power spectral density of the surface
        
        Findes the fft of the surface and stores it in your_instance.fft
        
        Parameters
        ----------
        (None)
        
        Returns
        -------
        (None), sets the psd attribute of the instance
            
        Examples
        --------
        
        >>my_surface.get_psd()
        sets the psd attribute of my_surface
        
        
        See Also
        --------
        get_fft
        get_acf
        show
        
        Notes
        -----
        Finds the psd by fouriertransforming the ACF, in doing so looks for the
        instance's acf property. if this is not found the acf is calculated and 
        set
        
        """
        # PSD is the fft of the ACF (https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density)
        if self.acf is None:
            self.get_acf()
        self.psd=self.get_fft(np.asarray(self.acf))
    
    def subtract_polynomial(self, order, profile_in=None): #Checked
        """ Flattens the surface by fitting and subtracting a polynomial
        
        Fits a polynomail to the surface the subtracts it from the surface, to
        remove slope or curve from imaging machines
        
        Parameters
        ----------
        
        order : int
            The order of the polynomial to be fitted
        profile_in : array-like optional
            if set the operation is perfomed on this profile rather than the 
            profile of the instance, the profile of the instance is not updated
            
        Returns
        -------
        adjusted : array
            The flattened profile
            
        Examples
        --------
        >>>my_surface.subtract_polynomail(2)
        Subtract a quadratic polynomial from the profile of my_surface
        the result is returned and the profile attribute is updated
        
        >>>flat_profile=my_surface.subtract_polynomial(2,my_surface.profile)
        Subtract a quadratic polynomial from the profile of my_surface and
        returns the result, the profile attribute is not updated
        
        >>>flat_profile_2=my_surface.subtract_polynomail(1, profile_2)
        Subtract a plae of best fit from profile_2 are return the result, the
        profile property of my_surface is not changed
        
        See Also
        --------
        birmingham
        numpy.linalg.lstsq
        
        Notes
        -----
        In principal polynomials of any integer order are supported
        
        """
        import itertools
        if profile_in is not None:
            profile=profile_in
        else:
            profile=self.profile
        x=range(profile.shape[0])
        y=range(profile.shape[1])
        Z=profile.flatten()
        X,Y=np.meshgrid(x,y)
        X=X.flatten()
        Y=Y.flatten()
        #fit polynomial
        n_cols=(order+1)**2
        G=np.zeros((Z.size, n_cols))
        ij=itertools.product(range(order+1), range(order+1))
        for k, (i,j) in enumerate(ij):
            G[:,k]=X**i*Y**j
        m, _, _, _ = np.linalg.lstsq(G, Z, rcond=None)
        
        poly=np.zeros_like(Z)
        
        ij=itertools.product(range(order+1), range(order+1))#must reset to itterate again
        
        for a, (i,j) in zip(m,ij):
            poly+=a*X**i*Y**j
        poly=poly.reshape(profile.shape)
        adjusted=profile-poly
        if profile_in is None:
            self.profile=adjusted
        return adjusted

    def birmingham(self, parameter_name, curved_surface=False, 
                   periodic_surface=False, profile_in=None, **kwargs): 
        """Find 3d surface roughness parameters
        
        Calculates and returns common surface roughness parameters also known 
        as birmingham parameters
        
        Parameters
        ----------
        parameter_name : str or list of str
            The name of the surface roughness parameter to be returned see note
        curved_surface : bool optional (False)
            True if the measurment surface was curved, see note
        periodic_surface : bool optioal (False)
            True if the surface is periodic
        profile_in : array-like optional (None)
            If a profile is supplied the parameters will be calculated on that 
            profile, limited support
        
        Returns
        -------
        out : float or list of float
            The requested parameters
        
        
        
        from: Stout, K., Sullivan, P., Dong, W., Mainsah, E., Luo, N., Mathia, 
        T., & Zahouani, H. (1993). 
        The development of methods for the characterisation of roughness in 
        three dimensions. EUR(Luxembourg), 358. 
        Retrieved from http://cat.inist.fr/?aModele=afficheN&cpsidt=49475
        chapter 12
        
        Returns the 3D surface parameters as defined in the above text:
        
        Before calculation the least squares plane is subtracted if a periodic
        surface is used this can be prevented by setting periodic_surface to 
        true. If a curved surface is used a bi quadratic polynomial is fitted
        and removed before analysis as descirbed in the above text. 
        
        If a list of valid parameter names is given this method will return a 
        list of parameter values.
        
        If a parameter based on summit descriptions is needed the key words:
            filter_cut_off (default False)
            and 
            four_nearest (default False) 
        can be set to refine what counts as a summit, see Surface.find_summits
        for more information. This is only used to find summits, calculations 
        are run on 'raw' surface.
        
        Descriptions of the parameters are given below.
            
        ** Amptitude parameters **
            Sq   - RMS deviation of surface height
            Sz   - Ten point height (based on definition of sumits)
            Ssk  - 'Skewness' of the surface (3rd moment)
            Sku  - 'Kurtosis' of the surface (4th moment)
        ** Spartial parameters ** 
            Sds  - Summit density, see note above on definition of summit
            Str  - Texture aspect ratio defined using the aacf
            Std  - Texture direction
            Sal  - Fastest decay auto corelation length
        ** hybrid parameters **
            Sdelq- RMS slope
            Ssc  - Mean summit curvature, see note above on definition of summit 
            Sdr  - Developed interfacial area ratio
        ** funcional parameters **
            Sbi  - Bearing index
            Sci  - Core fluid retention index - not implemented with equation 
                   book, appears to be incorrect, instead simple aproximation 
                   used, produces warning when used
            Svi  - Valley fluid retention index - see note above, produces 
                   produces warning when used
        ** non 'core' parameters (implemented) **
            Sa   - Mean amptitude of surface
            Stp  - Surface bearing ratio returns a listof curve points 
                   normalised as described in the above text
                   this is implemented without any interpolation
            Smr  - Material volume ratio of the surface required for 'sci', see
                   note above
            Svr  - Void volume ratio of the surface, as for previous
        ** non 'core' parameters (not implemented) **
            Sk   - Core roughness depth
            Spk  - Reduced summit height
            Svk  - Reduced valley depth
            Sr1  - Upper bearing area
            Sr2  - Lower bearing area
        
        examples:
            
        
        """
        #deafault p to self.profile
        p=profile_in
        if p is None:
            p=self.profile
        
        # recursive call to allow lists of parmeters to be retived at once
        
        if type(parameter_name) is list:
            out=[]
            for par_name in parameter_name:
                out.append(self.birmingham(par_name))
            return out
        else:
            parameter_name=parameter_name.lower()
        
        # First remove the base line, linear fit for flat surfaces or 
        # biquadratic polynomial for curved surfaces 
        
        if curved_surface:
            order=2
        else:
            order=1

        if periodic_surface:
            eta=p-np.mean(p.flatten())
        else:
            eta=self.subtract_polynomial(order,self.profile)
        
        # return parameter of interst 
        gs2=self._grid_spacing**2
        p_area=(self.profile.shape[0]-1)*(
                    self.profile.shape[1]-1)*gs2
        
        if parameter_name=='sq': #root mean square checked
            out=np.sqrt(np.mean(eta**2))
            
        elif parameter_name=='sa': #mean amptitude checked
            out=np.mean(np.abs(eta))
            
        elif parameter_name=='ssk': #skewness checked
            sq=np.sqrt(np.mean(eta**2))
            out=np.mean(eta**3)/sq**3
            
        elif parameter_name=='sku': #kurtosis checked
            sq=np.sqrt(np.mean(eta**2))
            out=np.mean(eta**4)/sq**4
            
        elif parameter_name in ['sds', 'sz', 'ssc']: # all that require sumits
            # summits is logical array of sumit locations
            kwargs['profile']=eta
            summits=self.find_summits(kwargs)
            if parameter_name=='sds': # summit density
                out=np.sum(summits)/(self._global_size[0]*self._global_size[1])
            elif parameter_name=='sz':
                kwargs['profile']=eta*-1
                valleys=self.find_summits(kwargs)
                summit_heights=eta[summits]
                valley_heights=eta[valleys]
                summit_heights=np.sort(summit_heights, axis=None)
                valley_heights=np.sort(valley_heights, axis=None)
                out=np.abs(valley_heights[:5]) +np.abs(summit_heights[-5:])/5
            else: # ssc mean summit curvature
                out=np.mean(self.get_summit_curvatures(summits, eta))
                
        elif parameter_name=='sdr': # developed interfacial area ratio 
            #ratio between actual surface area and projected or apparent 
            #surface area
            i_areas=[0.25*(((gs2+(eta[x,y]-eta[x,y+1])**2)**0.5+
                           (gs2+(eta[x+1,y+1]-eta[x+1,y])**2)**0.5)*
                          ((gs2+(eta[x,y]-eta[x+1,y])**2)**0.5+
                           (gs2+(eta[x,y+1]-eta[x+1,y+1])**2)**0.5))
                             for x in range(self.profile.shape[0]-1)
                             for y in range(self.profile.shape[1]-1)]
            i_area=sum(i_areas)        
            out=(i_area-p_area)/i_area
            
        elif parameter_name=='stp':
            # bearing area curve
            eta=eta/np.sqrt(np.mean(eta**2))
            heights=np.linspace(min(eta.flatten()),max(eta.flatten()),100)
            ratios=[np.sum(eta<height)/p_area for height in heights]
            out=[heights, ratios]
            
        elif parameter_name=='sbi': # bearing index
            index=int(eta.size/20) #TODO check this works should be betwween 0 and 1?
            sq=np.sqrt(np.mean(eta**2))
            out=sq/np.sort(eta.flatten())[index]
            
        elif parameter_name=='sci': # core fluid retention index
            sq=np.sqrt(np.mean(eta**2))
            index=int(eta.size*0.05)
            h005=np.sort(eta.flatten())[index]
            index=int(eta.size*0.8)
            h08=np.sort(eta.flatten())[index]
            
            V005=self.get_mat_or_void_volume_ratio(h005,True,eta,False)
            V08=self.get_mat_or_void_volume_ratio(h08,True,eta,False)
            
            out=(V005[0]-V08[0])/p_area/sq
            
        elif parameter_name=='svi': # valley fluid retention index
            sq=np.sqrt(np.mean(eta**2)) #TODO check this works, should it be between 0 and one?
            index=int(eta.size*0.8)
            h08=np.sort(eta.flatten())[index]
            V08=self.get_mat_or_void_volume_ratio(h08,True,eta,False)
            
            out=V08[0]/p_area/sq
            
        elif parameter_name=='str': # surface texture ratio
            if type(self.acf) is bool:
                self.get_acf()
            x=self._grid_spacing*np.arange(self.profile.shape[0]/-2,
                                       self.profile.shape[0]/2)
            y=self._grid_spacing*np.arange(self.profile.shape[1]/-2,
                                       self.profile.shape[1]/2)
            X,Y=np.meshgrid(x,y)
            distance_to_centre=np.sqrt(X**2+Y**2)
            min_dist=min(distance_to_centre[np.asarray(self.acf)<0.2])-1
            max_dist=max(distance_to_centre[np.asarray(self.acf)>0.2])

            out=min_dist/max_dist

        elif parameter_name=='std': # surface texture direction
            if type(self.fft) is bool:
                self.get_fft()
            apsd=self.fft*np.conj(self.fft)/p_area
            x=self._grid_spacing*np.arange(self.profile.shape[0])
            y=self._grid_spacing*np.arange(self.profile.shape[1])
            x=x-max(x)/2
            x=y-max(y)/2
            i,j = np.unravel_index(apsd.argmax(), apsd.shape)
            beta=np.arctan(i/j)
            if beta<(np.pi/2):
                out=-1*beta
            else:
                out=np.pi-beta
            
        elif parameter_name=='sal': # fastest decaying auto corelation length
            # shortest distance from center of ACF to point where R<0.2
            if type(self.acf) is bool:
                self.get_acf()
            x=self._grid_spacing*np.arange(self.profile.shape[0]/-2,
                                       self.profile.shape[0]/2)
            y=self._grid_spacing*np.arange(self.profile.shape[1]/-2,
                                       self.profile.shape[1]/2)
            X,Y=np.meshgrid(x,y)
            distance_to_centre=np.sqrt(X**2+Y**2)
            
            out=(min(distance_to_centre[np.asarray(self.acf)<0.2])-
                 self._grid_spacing/2)
        else:
            msg='Paramter name not recognised'
            raise ValueError(msg)
        return out
    
    def get_mat_or_void_volume_ratio(self, height, void=False, p=None,
                                     ratio=True, volume_ratio=None, 
                                     threshold=0.001):
        """ 
        Finds the material or void volume for a given plane height, uses an 
        approximation (that each point is a coloumn of material) original 
        formular didn't work. Produces warning to this effect.
        
        height  - the height of the cut off plane
        void    - if set to true the void volume will be calculated
        p       - default is self.profile other profiles can be provided if 
                  needed
        volume  - if provided the function will be 'reversed' and the height of
                  cut off plane which produces the desired volume will be 
                  returned, this is done by recursive calls to this function 
                  with binary searching. the process stops after the threshold 
                  has been reached
        threshold - the threshold used for binary searching of the height,
                    only used if volume is provided threshold is a fraction of 
                    total surface height range
        returns:
        
        the void or material volume ratio, the height.
        if 'max' or 'min' are supled as the height the actual volume will be 
        returned
        """
        msg=('get_mat_or_void_volume_ratio uses a simplified approximation, '
             'results will not be identical to those found with the formula'
             ' given in the original text, original formula dosn\'t seem to '
             'work')
        warnings.warn(msg)
        if p is None:
            p=self.profile
        p=p.flatten()
        # if a volume ratio is requested recursively call to find the height
        # for the given volume ratio
        max_height=max(p)
        min_height=min(p)
        
        if volume_ratio:
            accuracy=0.5
            position=0.5
            if volume_ratio<0 or volume_ratio>1:
                raise ValueError('Volume ratio must be bwteen 0 and 1')
            while accuracy>threshold:
                height=position*(max_height-min_height)+min_height
                vol, height=self.get_mat_or_void_volume_ratio(height, void, p)
                accuracy/=2
                if (vol>volume_ratio)!=void: #xor to reverse if void is needed
                    position-=accuracy
                else:
                    position+=accuracy
            return vol, height
        else: # 'normal' mode, volume ratio for specific height    
            n_pts=self.profile.shape[0]*self.profile.shape[1]
            total_vol=n_pts*(max_height-min_height)
            max_m=sum((p-min_height))
            m=sum([P-height for P in p if P>height])
            if void:
                all_above=(max_height-height)*n_pts
                void_out=all_above-m # void not below height
                m=total_vol-max_m-void_out
                if ratio:    
                    m=m/(total_vol-max_m)
            else:
                if ratio:
                    m=m/max_m
            return m, height

    def get_summit_curvatures(self, summits=False, p=None):
        if p is None:
            p=self.profile
        if type(summits) is bool:
            summits=self.find_summits(profile=p)
        verts=np.transpose(np.nonzero(summits))
        curves= [-0.5*(p[vert[0]-1,vert[1]]+p[vert[0]+1,vert[1]]+p[vert[0],
                        vert[1]-1]+p[vert[0],vert[1]+1]-4*p[vert[0],vert[1]]
                )/self._grid_spacing**2 for vert in verts]
        return curves
        
        
    def find_summits(self, *args, **kwargs):
        #TODO make work for periodic surface
        # summits are found by low pass filtering at the required scale then 
        # finding points which are higher than 4 or 8 nearest neigbours
        if args[0] is dict:    
            kwargs={**args[0], **kwargs}
        
        if 'profile' in kwargs:
            profile=kwargs['profile']
        else:
            profile=self.profile
            
        if 'filter_cut_off' in kwargs:
            filtered_profile=self.low_pass_filter(kwargs['filter_cut_off'], 
                                                  profile, False)
        else:
            filtered_profile=self.profile
        summits=np.ones(self.profile[1:-1,1:-1].shape, dtype=bool)
        if 'four_nearest' in kwargs and not kwargs['four_nearest']:
            x=[-1,+1,0,0,-1,-1,+1,+1]
            y=[0,0,-1,+1,-1,+1,-1,+1]
        else:
            x=[-1,+1,0,0]
            y=[0,0,-1,+1]
        
        for i in range(len(x)):   
            summits=np.logical_and(summits,(filtered_profile[1:-1,1:-1]>
                              filtered_profile[1+x[i]:-1+x[i] or 
                                               None,1+y[i]:-1+y[i] or None]))
        #pad summits with falses
        summits=np.pad(summits, 1, 'constant', constant_values=False)
        return summits
        
    def low_pass_filter(self, cut_off_freq, profile=False, copy=False):
        import scipy.signal
        if not profile:
            profile=self.profile
        sz=self.profile.shape
        x=np.arange(1, sz[0]+1)
        y=np.arange(1, sz[1]+1)
        X,Y=np.meshgrid(x,y)
        D=np.sqrt(X**2+Y**2)
        ws=2*np.pi/self._grid_spacing
        wc=cut_off_freq*2*np.pi
        h=(wc/ws)*scipy.special.j1(2*np.pi*(wc/ws)*D)/D
        filtered_profile=scipy.signal.convolve2d(profile,h,'same')
        if copy:
            return filtered_profile
        else:
            self.profile=filtered_profile
            return
    
    def read_from_file(self, path, **kwargs):
        """ 
        #TODO add documentation to this
        kwargs can be header lines etc, args is kwargs from init
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
            self.set_grid_spacing(data['Header']['PixelSizeXMeter'])
        else:
            msg=('Path does not have a recognised file extention')
            raise ValueError(msg)
        self.is_descrete=True
        return kwargs
    
    def resample(self, new_grid_spacing, return_profile=False, 
                 remake_interpolator=False):
        if remake_interpolator or not self._inter_func:
            import scipy.interpolate
            x0=np.arange(0, self._global_size[0], self._grid_spacing)
            y0=np.arange(0, self._global_size[1], self._grid_spacing)
            self._inter_func=scipy.interpolate.RectBivariateSpline(x0, y0, 
                                                                  self.profile)
        x1=np.arange(0, self._global_size[0], new_grid_spacing)
        y1=np.arange(0, self._global_size[1], new_grid_spacing)
        inter_profile=self._inter_func(x1,y1)
        
        if return_profile:
            return inter_profile
        else:
            self.profile=inter_profile
            self.set_grid_spacing(new_grid_spacing)
        
    def fill_holes(self, hole_value='auto', mk_copy=False, remove_boarder=True, 
                   b_thresh=0.99):
        """
        Replaces specificed values with filler
        
        Uses biharmonic equations algorithm to fill holes 
        
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
        
        Notes
        -----
        When alicona images are imported the invalid pixel value is 
        automatically set to nan so this will work in auto mode
        
        Holes are filled with bi harmonic equations
        
        See Also
        --------
        skimage.restoration.inpaint.inpaint_biharmonic
        
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
        
        if not self._grid_spacing:
            if self._global_size:
                self.set_global_size(self._global_size)
                new_gs=self._grid_spacing
            else:
                new_gs=None
        else:
            new_gs=self._grid_spacing
            
        
        if mk_copy:
            new_surf=Surface(profile=profile_out, 
                             grid_spacing=new_gs,
                             is_descrete=self.is_descrete)
            return new_surf
        else:
            self.profile=profile_out
            self.set_grid_spacing(new_gs)
            return
        
        
    def __add__(self, other):
        if self._global_size==other._global_size and self._global_size:
            if self._grid_spacing==other._grid_spacing:
                out=Surface(profile=self.profile+other.profile, 
                            grid_spacing=self._grid_spacing, 
                            is_descrete=self.is_descrete)
                return out
            else:
                msg="Surface sizes do not match: resampling"
                warnings.warn(msg)
                ## resample surface with coarser grid then add again
                if self._grid_spacing>other.grid_spacing:
                    self.resample(other.grid_spacing, False)
                else:
                    other.resample(self._grid_spacing)
                    
                return self+other
            
        elif self.profile.shape==other.profile.shape:
            if self._grid_spacing:
                if other._grid_spacing:
                    if self._grid_spacing==other._grid_spacing:
                        gs=self._grid_spacing
                    else:    
                        msg=('Surfaces have diferent sizes and grid spacing is' 
                             ' set for both for element wise adding unset the '
                             'grid spacing for one of the surfaces using: '
                             'set_grid_spacing(None) before adding')
                        raise AttributeError(msg)
                else: #only self not other
                    gs=self._grid_spacing
            else: #not self
                gs=other._grid_spacing
                
            out=Surface(profile=self.profile+other.profile,
                            grid_spacing=gs, 
                            is_descrete=self.is_descrete)
            return out
        else:
            raise ValueError('surfaces are not compatible sizes cannot add')
            
    def __sub__(self, other):
        if self._global_size==other._global_size and self._global_size:
            if self._grid_spacing==other._grid_spacing:
                out=Surface(profile=self.profile-other.profile, 
                            grid_spacing=self._grid_spacing, 
                            is_descrete=self.is_descrete)
                return out
            else:
                msg="Surface sizes do not match: resampling"
                warnings.warn(msg)
                ## resample surface with coarser grid then add again
                if self._grid_spacing>other.grid_spacing:
                    self.resample(other.grid_spacing, False)
                else:
                    other.resample(self._grid_spacing)
                    
                return self-other
            
        elif self.profile.shape==other.profile.shape:
            if self._grid_spacing:
                if other._grid_spacing:
                    if self._grid_spacing==other._grid_spacing:
                        gs=self._grid_spacing
                    else:    
                        msg=('Surfaces have diferent sizes and grid spacing is' 
                             ' set for both for element wise adding unset the '
                             'grid spacing for one of the surfaces using: '
                             'set_grid_spacing(None) before subtracting')
                        raise AttributeError(msg)
                else: #only self not other
                    gs=self._grid_spacing
            else: #not self
                gs=other._grid_spacing
                
            out=Surface(profile=self.profile-other.profile,
                            grid_spacing=gs, 
                            is_descrete=self.is_descrete)
            return out
        else:
            raise ValueError('surfaces are not compatible sizes cannot'
                             ' subtract')
        
    def show(self, property_to_plot='profile', plot_type='default', ax=False,
             *args):
        """ 
        Method for plotting anything of interest for the surface
        
        If a list of properties is provided all will be plotted on separate 
        sub plots of the same (new) figure, if a list of plot types is also
        provided these will be used as the types, if a single type is provided 
        this will be used for all applicable plots, if the list is too short it
        will be extended to the same length as the list of properties with 
        the default options for the polts.
        
        output : a list of axis handles
        
        Valid propertie to plot are:
            ** 2D types **
            profile
            fft2D
            psd
            acf
            ** 1D types **
            histogram
            fft1D
            disthist - requires seaborn histogram of data with dist fitted
            QQ - name of distribution must also be provided as *args 
            ** other **
            other can be given if *args contains 2 or 3 lists of items a 1 or 2
            dimentional plot of type surface or line will be plotted, this is 
            primarily for debugging and options are limited
        Valid plot types and the default type depend on the property_to_plot:
            ** for 2D types **
            surface - default
            image
            image_wrap - image plot but wraps corners to centre common for fft
            mesh
            ** for 1D types **
            bar - default for histogram
            line - default for fft1D
            scatter
            area
            
        example:
            self.show(['fft2D','fft2D','fft2D'], ['mesh', 'image', 'default'])
            shows the 2D fft of the surface profile with a range of plot types
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
                x=self._grid_spacing*np.arange(self.profile.shape[0])
                y=self._grid_spacing*np.arange(self.profile.shape[1])
                
            elif property_to_plot=='fft2d':
                labels=['Fourier transform of surface', 'u', 'v', '|F(x)|']
                if self.fft is None:
                    self.get_fft()
                Z=np.abs(np.fft.fftshift(self.fft))
                x=np.fft.fftfreq(self.profile.shape[0], self._grid_spacing)
                y=np.fft.fftfreq(self.profile.shape[1], self._grid_spacing)
                
            elif property_to_plot=='psd':
                labels=['Power spectral density', 'u', 'v', 'Power/ frequency']
                if self.psd is None:
                    self.get_psd()
                Z=np.abs(np.fft.fftshift(self.psd))
                x=np.fft.fftfreq(self.profile.shape[0], self._grid_spacing)
                y=np.fft.fftfreq(self.profile.shape[1], self._grid_spacing)
                
            elif property_to_plot=='acf':
                labels=['Auto corelation function', 'x', 'y', 
                        'Surface auto correlation']
                if self.acf is None:
                    self.get_acf()
                Z=np.abs(np.asarray(self.acf))
                x=self._grid_spacing*np.arange(self.profile.shape[0])
                y=self._grid_spacing*np.arange(self.profile.shape[1])
                x=x-max(x)/2
                y=y-max(y)/2
            elif property_to_plot=='apsd':
                labels=['Angular power spectral density', 'x', 'y']
                if self.fft is None:
                    self.get_fft()
                p_area=(self.profile.shape[0]-1)*(
                    self.profile.shape[1]-1)*self.grid_spacing**2
                Z=self.fft*np.conj(self.fft)/p_area
                x=self._grid_spacing*np.arange(self.profile.shape[0])
                y=self._grid_spacing*np.arange(self.profile.shape[1])
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
                if args:
                    ax.plot_wireframe(X, Y, np.transpose(Z), rstride=args[0], 
                                      cstride=args[1])
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
                    x=np.fft.fftfreq(self.profile.shape[0], 
                                     self._grid_spacing)
                    y=np.abs(self.fft/self.profile.shape[0])
                    # line plot for 1d surfaces
                    ax.plot(x,y)
                    ax.xlim(0,max(x))
                else:
                    labels=['Scatter of frequency magnitudes', 
                            'frequency', '|F(x)|']
                    u=np.fft.fftfreq(self.profile.shape[0], 
                                     self._grid_spacing)
                    v=np.fft.fftfreq(self.profile.shape[1], 
                                     self._grid_spacing)
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
                if args:
                    seaborn.distplot(self.profile.flatten(), fit=args[0], 
                                 kde=False, ax=ax)
                else:
                    seaborn.distplot(self.profile.flatten(), ax=ax)    
            elif property_to_plot=='qq':
                from scipy.stats import probplot
                labels=['Probability plot', 'Theoretical quantities', 
                        'Ordered values']
                if args:
                    probplot(self.profile.flatten(), dist=args[0], fit=True,
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
        return np.asarray(self.profile)

    def _get_points_from_extent(self, extent, grid_spacing):
        if type(grid_spacing) in [int, float, np.float32, np.float64]:
            grid_spacing=[grid_spacing]*2
        if type(extent[0]) in [int, float, np.float32, np.float64]:
            extent=[[0,extent[0]],[0,extent[1]]]
        x=np.arange(extent[0][0], extent[0][1], grid_spacing[0])
        y=np.arange(extent[1][0], extent[1][1], grid_spacing[1])
        
        X,Y=np.meshgrid(x,y)
        
        return(X,Y)
    
    def rotate(radians):
        """
        rotate the surface relative to the grid and reinterpolate
        """
        raise NotImplementedError('Not implemented yet')
    
    def _descretise_checks(self):
        if self.is_descrete:
            msg=('Surface is already discrete this will overwrite surface'
                 ' profile')
            raise warnings.warn(msg)
        try: 
            grid_spacing=self._grid_spacing
        except AttributeError:
            msg='A grid spacing must be provided before descretisation'
            raise AttributeError(msg)
        try:
            pts_each_direction=[int(gs/grid_spacing) for gs in 
                                    self._global_size]
            total_pts=1
            for pts in pts_each_direction:
                total_pts *= pts
            self.total_pts=total_pts
            self._pts_each_direction=pts_each_direction
        except AttributeError:
            msg='Global size must be set before descretisation'
            raise AttributeError(msg)
        if total_pts>10E7:
            warnings.warn('surface contains over 10^7 points calculations will'
                          ' be slow, consider splitting surface or analysis')
        return    
    
    
if __name__=='__main__':
    A=Surface(file_name='C:\\Users\\44779\\code\\SlipPY\\image1_no header_units in nm', file_type='csv', delimitor=' ', grid_spacing=0.001)
    #testing show()
    #types2d=['profile', 'fft2d', 'psd', 'acf', 'apsd']
    #types1d=['histogram','fft1d', 'qq', 'disthist']
    #A.show(['profile','fft2d','acf'], ['surface', 'mesh', 'image'])
    #A.show(['histogram', 'fft1d', 'qq', 'disthist'])
    #out=A.birmingham(['sa', 'sq', 'ssk', 'sku'])
    #out=A.birmingham(['sds','sz', 'ssc'])
    #out=A.birmingham(['std','sbi', 'svi', 'str'])
    #out=A.birmingham(['sdr', 'sci', 'str'])
    