import numpy as np
import slippy.surface as S

surf = S.FlatSurface(grid_spacing=1, extent=[9,9])
surf.descretise()
profile=surf.profile
grid_spacing=surf.grid_spacing
depth=5
parameters={}


#def _full(profile, grid_spacing, parameters, depth):
"""
Meshes a surface with hexahedral elements in a grid

Parameters
----------


valid keys in parameters:
    max_aspect
    min_aspect
    mode {linear, exponential}
"""
valid_modes=['linear', 'exponential']
aspect=[1,3]
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

h_min=min(profile.flatten())
profile=profile-h_min+depth

h_min=depth
h_max=max(profile.flatten())

if aspect[0]>aspect[1]:
    seg_max_top=aspect[0]*grid_spacing
    seg_max_bottom=aspect[1]*grid_spacing/h_min*h_max
else:
    seg_max_top=aspect[0]*grid_spacing/h_min*h_max
    seg_max_bottom=aspect[1]*grid_spacing

n_segs=int(np.ceil(h_max/(seg_max_top+seg_max_bottom)*2))
segs_raw=np.cumsum(np.linspace(seg_max_bottom, seg_max_top, n_segs))

# normalised y coodinates for each node multiply by the hieght of the 
# surface to get the actual y coordinates
segs_norm=np.insert(segs_raw, 0, 0)/max(segs_raw)

n_pts=profile.size

#just index X and Y using mod operator
X, Y = np.meshgrid(np.arange(profile.shape[0])*grid_spacing,
                   np.arange(profile.shape[1])*grid_spacing)
Z=np.reshape(np.repeat(segs_norm, n_pts), (n_segs+1, 
  profile.shape[0], profile.shape[1]))*profile

# got all the nodes now construct the elements


#import numpy as np
#import itertools
#from slippy.surface import Surface, ACF
#from pytest import raises as assert_raises
#import scipy.signal
#import warnings
#
#def roughness(profile_in, parameter_name, grid_spacing=float('inf'), mask=None, 
#              curved_surface=False, no_flattening=False, filter_cut_off=False,
#              four_nearest=False): 
#    """Find 3d surface roughness parameters
#    
#    Calculates and returns common surface roughness parameters also known 
#    as birmingham parameters
#    
#    Parameters
#    ----------
#    profile_in : array like or Surface
#        The surface profile or surface object to be used
#    parameter_name : str or list of str
#        The name of the surface roughness parameter to be returned see notes 
#        for descriptions of each
#    grid_spacing : float 
#        The distance between adjacent grid points in the surface
#        only required for some parameters, see notes    
#    mask : array-like same shape as profile or float (None)
#        If an array, the array is used as a mask for the profile, it must be  
#        the same shape as the profile, if a float or list of floats is given, 
#        those values are excluded from the calculation. If None, no mask is 
#        used. Limited applicability, see notes
#    curved_surface : bool optional (False)
#        True if the measurment surface was curved, in this case a 2nd order
#        polynomial is subtracted other wise a 1st order polynomial is 
#        subtracted before the measuremt
#    no_flattening : bool optional (False)
#        If true, flattening will be skipped, no polynomial will be subtracted
#        before calculation of parameters, used for periodic surfaces or to 
#        save time
#        
#    
#    Returns
#    -------
#    out : float or list of floats
#        The requested parameters
#        
#    Other parameters
#    ----------------
#    four_nearest : bool optional (False) 
#        If true any point that is higher than it's 4 nearest neigbours will be 
#        counted as a summit, otherwise a point must be higher than it's 8 
#        nearest neigbours to be a summit. Only used if summit descriptions
#        are required, passed to find_summits.
#    filter_cut_off : float optional (None)
#        If given the surface will be low pass filtered before finding sumits. 
#        Only used if summit descriptions are required, passed to find_summits.
#        
#    See Also
#    --------
#    Surface : a helper class with useful surface analysis functionality
#    subtract_polynomial
#    find_summits
#    get_mat_or_void_ratio
#    get_summit_curvatures
#    
#    Notes
#    -----
#    From: Stout, K., Sullivan, P., Dong, W., Mainsah, E., Luo, N., Mathia, 
#    T., & Zahouani, H. (1993). 
#    The development of methods for the characterisation of roughness in 
#    three dimensions. EUR(Luxembourg), 358. 
#    Retrieved from http://cat.inist.fr/?aModele=afficheN&cpsidt=49475
#    chapter 12
#    
#    Before calculation the least squares plane is subtracted if a periodic
#    surface is used this can be prevented by setting the no_flattening key 
#    word to true. If a curved surface is used a bi quadratic polynomial is 
#    fitted and removed before analysis as descirbed in the above text. 
#    
#    If a list of valid parameter names is given this method will return a 
#    list of parameter values.
#    
#    If a parameter based on summit descriptions is needed the key words:
#        filter_cut_off (default False)
#        and 
#        four_nearest (default False) 
#    can be set to refine what counts as a summit, see find_summits
#    for more information. This is only used to find summits, calculations 
#    are run on 'raw' surface.
#    
#    Descriptions of each of the surface roughness parameters are given below
#    
#    * masking supported
#    + requires grid_spacing
#    - requires grid spacing only if filtering is used for summit definition
#    
#    Summit parameters only support masking if low pass filtering is not 
#    required
#    
#    Amptitude parameters
#        Sq   - RMS deviation of surface height *
#        Sz   - Ten point height (based on definition of sumits) *-
#        Ssk  - Skew of the surface (3rd moment) *
#        Sku  - Kurtosis of the surface (4th moment) *
#    Spartial parameters 
#        Sds  - Summit density*-, see note above on definition of summit
#        Str  - Texture aspect ratio defined using the aacf 
#        Std  - Texture direction
#        Sal  - Fastest decay auto corelation length +
#    hybrid parameters
#        Sdelq- RMS slope +
#        Ssc  - Mean summit curvature, see note above on definition of summit *+ 
#        Sdr  - Developed interfacial area ratio +
#    funcional parameters
#        Sbi  - Bearing index *
#        Sci  - Core fluid retention index *
#        Svi  - Valley fluid retention index *
#    non 'core' parameters (implemented)
#        Sa   - Mean amptitude of surface *
#        Stp  - Surface bearing ratio returns a list of curve points 
#               normalised as described in the above text
#               this is implemented without any interpolation *
#        Smr  - Material volume ratio of the surface required for 'sci', see
#               note above *
#        Svr  - Void volume ratio of the surface, as for previous *
#    non 'core' parameters (not implemented)
#        Sk   - Core roughness depth
#        Spk  - Reduced summit height
#        Svk  - Reduced valley depth
#        Sr1  - Upper bearing area
#        Sr2  - Lower bearing area
#    
#    Parameter names are not case sensitive
#    
#    Examples
#    --------
#    
#    
#    """
#    
#    if profile_in is Surface or issubclass(type(profile_in), Surface):
#        if not grid_spacing:
#            try:
#                grid_spacing=profile_in._grid_spacing
#            except AttributeError:
#                pass
#    
#    profile=np.asarray(profile_in) # to make sure it's not overwritten by masking
#    
#    if mask is not None:
#        if type(mask) is float:
#            if np.isnan(mask):
#                mask=~np.isnan(profile)
#            else:
#                mask=~profile==mask
#        else:
#            mask=np.asarray(mask, dtype=bool)
#            if not mask.shape==profile.shape:
#                msg=("profile and mask shapes do not match: profile is"
#                    "{profile.shape}, mask is {mask.shape}".format(**locals()))
#                raise TypeError(msg)
#    else:
#        mask=np.ones_like(profile, dtype=bool)
#    
#    #first subtract polynomial
#    if curved_surface:
#        order=2
#    else:
#        order=1
#
#    if no_flattening:
#        eta=profile
#    else:
#        eta=subtract_polynomial(profile, order, mask)
#    
#    # recursive call to allow lists of parmeters to be retived at once
#    if type(parameter_name) is list:
#        out=[]
#        for par_name in parameter_name:
#            out.append(roughness(eta, par_name, grid_spacing=grid_spacing,
#                                 mask=mask, no_flattening=True, 
#                                 filter_cut_off=filter_cut_off,
#                                 four_nearest=four_nearest))
#        return out
#    else:
#        try:
#            parameter_name=parameter_name.lower()
#        except AttributeError:
#            msg=("Parameters must be strings or list of strings")
#            raise ValueError(msg) 
#    
#    # return parameter of interst 
#    global_size=[grid_spacing*dim for dim in profile.shape]
#    gs2=grid_spacing**2
#    eta_masked=eta[mask]
#    num_pts_m=eta_masked.size
#    p_area_m=num_pts_m*gs2
#    
#    p_area_t=eta.size*gs2
#    
#    if parameter_name=='sq': #root mean square checked
#        out=np.sqrt(np.mean(eta_masked**2))
#        
#    elif parameter_name=='sa': #mean amptitude checked
#        out=np.mean(np.abs(eta_masked))
#        
#    elif parameter_name=='ssk': #skewness checked
#        sq=np.sqrt(np.mean(eta_masked**2))
#        out=np.mean(eta_masked**3)/sq**3
#        
#    elif parameter_name=='sku': #kurtosis checked
#        sq=np.sqrt(np.mean(eta_masked**2))
#        out=np.mean(eta_masked**4)/sq**4
#    
#    elif parameter_name in ['sds', 'sz', 'ssc']: # all that require sumits
#        # summits is logical array of sumit locations
#        summits=find_summits(eta, grid_spacing, mask, four_nearest, 
#                             filter_cut_off)
#        if parameter_name=='sds': # summit density
#            out=np.sum(summits)/(num_pts_m)
#        elif parameter_name=='sz':
#            valleys=find_summits(-1*eta, grid_spacing, mask, four_nearest, 
#                             filter_cut_off)
#            summit_heights=eta[summits]
#            valley_heights=eta[valleys]
#            summit_heights=np.sort(summit_heights, axis=None)
#            valley_heights=np.sort(valley_heights, axis=None)
#            out=np.abs(valley_heights[:5])+np.abs(summit_heights[-5:])/5
#        else: # ssc mean summit curvature
#            out=np.mean(get_summit_curvatures(eta, summits, grid_spacing))
#            
#    elif parameter_name=='sdr': # developed interfacial area ratio 
#        #ratio between actual surface area and projected or apparent 
#        #surface area
#        i_areas=[0.25*(((gs2+(eta[x,y]-eta[x,y+1])**2)**0.5+
#                       (gs2+(eta[x+1,y+1]-eta[x+1,y])**2)**0.5)*
#                      ((gs2+(eta[x,y]-eta[x+1,y])**2)**0.5+
#                       (gs2+(eta[x,y+1]-eta[x+1,y+1])**2)**0.5)) 
#                        for x in range(eta.shape[0]-1) 
#                        for y in range(eta.shape[1]-1)]
#        i_area=sum(i_areas)        
#        out=(i_area-p_area_t)/i_area
#        
#    elif parameter_name=='stp':
#        # bearing area curve
#        eta_rel=eta_masked/np.sqrt(np.mean(eta_masked**2))
#        heights=np.linspace(min(eta_rel),max(eta_rel),100)
#        ratios=[np.sum(eta_masked<height)/p_area_m for height in heights]
#        out=[heights, ratios]
#        
#    elif parameter_name=='sbi': # bearing index
#        index=int(eta_masked.size/20)
#        sq=np.sqrt(np.mean(eta_masked**2))
#        out=sq/np.sort(eta_masked)[index]
#        
#    elif parameter_name=='sci': # core fluid retention index
#        sq=np.sqrt(np.mean(eta_masked**2))
#        eta_m_sorted=np.sort(eta_masked)
#        index=int(eta_masked.size*0.05)
#        h005=eta_m_sorted[index]
#        index=int(eta_masked*0.8)
#        h08=eta_m_sorted[index]
#        
#        V005=get_mat_or_void_volume_ratio(h005,eta,void=True,mask=mask)
#        V08=get_mat_or_void_volume_ratio(h08,eta,void=True,mask=mask)
#        
#        out=(V005-V08)/p_area_m/sq
#        
#    elif parameter_name=='svi': # valley fluid retention index
#        sq=np.sqrt(np.mean(eta_masked**2))
#        index=int(eta_masked.size*0.8)
#        h08=np.sort(eta_masked)[index]
#        V08=get_mat_or_void_volume_ratio(h08,eta,void=True,mask=mask)
#        out=V08/p_area_m/sq
#        
#    elif parameter_name=='str': # surface texture ratio
#        
#        acf=np.asarray(ACF(eta))
#        
#        x=np.arange(eta.shape[0]/-2,eta.shape[0]/2)
#        y=np.arange(eta.shape[1]/-2,eta.shape[1]/2)
#        X,Y=np.meshgrid(x,y)
#        distance_to_centre=np.sqrt(X**2+Y**2)
#        min_dist=min(distance_to_centre[acf<0.2])-0.5
#        max_dist=max(distance_to_centre[acf>0.2])+0.5
#
#        out=min_dist/max_dist
#
#    elif parameter_name=='std': # surface texture direction
#        fft=np.fft.fft2(eta)
#        
#        apsd=fft*np.conj(fft)/p_area_t
#        x=np.arange(eta.shape[0]/-2,eta.shape[0]/2)
#        y=np.arange(eta.shape[1]/-2,eta.shape[1]/2)
#        i,j = np.unravel_index(apsd.argmax(), apsd.shape)
#        beta=np.arctan(i/j)
#        
#        if beta<(np.pi/2):
#            out=-1*beta
#        else:
#            out=np.pi-beta
#        
#    elif parameter_name=='sal': # fastest decaying auto corelation length
#        # shortest distance from center of ACF to point where R<0.2
#        acf=np.asarray(ACF(eta))
#        
#        x=grid_spacing*np.arange(eta.shape[0]/-2,
#                                   eta.shape[0]/2)
#        y=grid_spacing*np.arange(eta.shape[1]/-2,
#                                   eta.shape[1]/2)
#        X,Y=np.meshgrid(x,y)
#        
#        distance_to_centre=np.sqrt(X**2+Y**2)
#        
#        out=min(distance_to_centre[acf<0.2])
#        
#    else:
#        
#        msg='Paramter name not recognised'
#        raise ValueError(msg)
#    
#    return out
#
#def get_height_of_mat_or_void_ratio(ratio, profile, void=False, mask=None, 
#                                       accuracy=0.001):
#    """Finds the cut off height of a specified material or void volume ratio
#    
#    Parameters
#    ----------
#    ratio : float {from 0 to 1}
#        the target material or void volume ratio
#    profile : 2D array-like or Surface object
#        The surface profile to be used in the calculation
#    void : bool optional (False)
#        If set to true the height for the void volume ratio will be calculated
#        otherwise the height for the material volume ratio will be calculated
#    mask : array-like (bool) same shape as profile or float (defaults to None)
#        If an array, the array is used as a mask for the profile, must be the 
#        same shape as the profile, if a float is given, values which match are
#        excluded from the calculation 
#    accuracy : float optional (0.0001)
#        The threshold value to stop itterations
#
#    
#    Returns
#    -------
#    height : float
#        the height at which the input surface has the specified material or 
#        void ratio
#    
#    See also
#    --------
#    get_mat_or_void_volume_ratio
#    roughness
#    subtract_polynomial
#    
#    Notes
#    -----
#    This function should not be used without first flattening the surface using
#    subtract_polynomial
#    
#    This function uses a simplified algorithim assuming that each point in the
#    surface can be modeled as a column of material.
#    
#    Examples
#    --------
#    
#    """
#    import scipy.optimize
#    
#    p=np.asarray(profile)
#        
#    if mask is not None:
#        if type(mask) is float:
#            if np.isnan(mask):
#                mask=~np.isnan(p)
#            else:
#                mask=~p==mask
#        else:
#            mask=np.asarray(mask, dtype=bool)
#            if not mask.shape==p.shape:
#                msg=("profile and mask shapes do not match: profile is"
#                    "{p.shape}, mask is {mask.shape}".format(**locals()))
#                raise TypeError(msg)
#        
#        p=p[~mask]
#    else:
#        p=p.flatten()
#    
#    min_h=min(p)
#    max_h=max(p)
#    
#    if void:
#        first_guess=min_h+ratio*(max_h-min_h)
#    else:
#        first_guess=max_h-ratio*(max_h-min_h)
#    
#    min_func=lambda h: (get_mat_or_void_volume_ratio(h, p, void)-ratio)**2
#    
#    output=scipy.optimize.minimize(min_func, first_guess, bounds=(min_h,max_h),
#                                   tol=accuracy)
#    
#    height=output.x[0]
#    
#    return height
#
#def get_mat_or_void_volume_ratio(height, profile, void=False, mask=None,
#                                 ratio=True, grid_spacing=None):
#    """ Finds the material or void volume ratio
#    
#    Finds the material or void volume for a given plane height, uses an 
#    approximation (that each point is a column of material)
#    
#    Parameters
#    ----------
#    profile : 2D array-like or Surface object
#        The surface profile to be used in the calculation
#    height : float
#        The height of the cut off plane
#    void : bool optional (False)
#        If set to true the void volume will be calculated otherwise the 
#        material volume is calculated
#    mask : array-like (bool) same shape as profile or float (defaults to None)
#        If an array, the array is used as a mask for the profile, must be the 
#        same shape as the profile, if a float is given, values which match are
#        excluded from the calculation 
#    ratio : bool optional (True)
#        If true the material or void ratio will be returned, if false the
#        absolute value will be returned, this requires the grid_spacing
#        keyword to be set
#    grid_spacing : float 
#        The distance between adjacent grid points in the surface
#
#    
#    Returns
#    -------
#    out : float
#        The requested output parameter
#    
#    See also
#    --------
#    get_height_of_mat_or_void_ratio
#    roughness
#    subtract_polynomial
#    
#    Notes
#    -----
#    This function should not be used without first flattening the surface using
#    subtract_polynomial
#    
#    This function uses a simplified algorithim assuming that each point in the
#    surface can be modeled as a column of material.
#    
#    
#    Examples
#    --------
#    
#   
#    """
#    
#    p=np.asarray(profile)
#    
#    if profile is Surface or issubclass(type(profile), Surface):
#        if not grid_spacing and not ratio:
#            try:
#                grid_spacing=profile._grid_spacing
#            except AttributeError:
#                pass
#    
#    if not grid_spacing and not ratio:
#        msg=("Grid spacing keyword or property of input surface must be set "
#             "for absoulte results, see Surface.set_grid_spacing if you are"
#             " using surface objects")
#        raise ValueError(msg)
#        
#    if mask is not None:
#        if type(mask) is float:
#            if np.isnan(mask):
#                mask=~np.isnan(p)
#            else:
#                mask=~p==mask
#        else:
#            mask=np.asarray(mask, dtype=bool)
#            if not mask.shape==p.shape:
#                msg=("profile and mask shapes do not match: profile is"
#                    "{p.shape}, mask is {mask.shape}".format(**locals()))
#                raise TypeError(msg)
#        
#        p=p[~mask]
#    else:
#        p=p.flatten()
#        
#    max_height=max(p)
#    min_height=min(p)
#    
#    n_pts=p.size
#    total_vol=n_pts*(max_height-min_height)
#    max_m=sum(p-min_height)
#    
#    material=sum(p-height)*(p>height)
#    if void:
#        all_above=(max_height-height)*n_pts
#        void_out=all_above-material # void not below height
#        void=total_vol-max_m-void_out
#        if ratio:    
#            out=void/(total_vol-max_m)
#        else:
#            out=void*grid_spacing**3
#    else:
#        if ratio:
#            out=material/max_m
#        else:
#            out=material*grid_spacing**3
#    return out
#
#def get_summit_curvatures(profile, summits=None, grid_spacing=None, mask=None,
#                          filter_cut_off=None, four_nearest=False):
#    """ find the curvatures of the sumsits
#    
#    Parameters
#    ----------
#    profile : N by M array-like or Surface object
#        The surface profile for analysis
#    summits : N by M array (optional)
#        A bool array True at the location of the summits, if not supplied the 
#        summits are found using find_summits first, see notes
#    grid_spacing : float optional (False)
#        The distanc between points on the grid of the surface profile. Required
#        only if the filter_cut_off is set and profile is not a surface object
#    mask : array-like (bool)N by M or float optional (None)
#        If an array, the array is used as a mask for the profile, must be the 
#        same shape as the profile, if a float is given, values which match are
#        excluded from the calculation 
#    Returns
#    -------
#    curves : array
#        Array of summit curvatures of size sum(summits.flatten())
#    
#    Other parameters
#    ----------------
#    four_nearest : bool optional (False) 
#        If true any point that is higher than it's 4 nearest neigbours will be 
#        counted as a summit, otherwise a point must be higher than it's 8 
#        nearest neigbours to be a summit. Only used is summits are not given.
#    filter_cut_off : float optional (None)
#        If given the surface will be low pass filtered before finding sumits. 
#        Only used if summits are not given
#    
#    See also
#    --------
#    find_summits
#    roughness
#    
#    Notes
#    -----
#    If the summits parameter is not set, any key word arguments that can be 
#    passed to find_summits can be passed through this function.  
#    
#    Examples
#    --------
#    #TODO
#    """
#    if profile is Surface or issubclass(type(profile), Surface):
#        if not grid_spacing:
#            try:
#                grid_spacing=profile._grid_spacing
#            except AttributeError:
#                pass
#        profile=profile.profile
#    
#    gs2=grid_spacing**2
#    
#    if summits is None:
#        summits=find_summits(profile, filter_cut_off=filter_cut_off,
#                             grid_spacing=grid_spacing, 
#                             four_nearest=four_nearest, mask=mask)
#    verts=np.transpose(np.nonzero(summits))
#    curves= [-0.5*(profile[vert[0]-1,vert[1]]+profile[vert[0]+1,vert[1]]+
#                   profile[vert[0],vert[1]-1]+profile[vert[0],vert[1]+1]
#                   -4*profile[vert[0],vert[1]])/gs2 for vert in verts]
#    return curves
#    
#    
#def find_summits(profile, grid_spacing=False, mask=None, 
#                 four_nearest=False, filter_cut_off=None):
#    """ Finds highpoints after low pass filtering
#    
#    Parameters
#    ----------
#    profile : N by M array-like
#        The surface profile for analysis
#    grid_spacing : float optional (False)
#        The distanc between points on the grid of the surface profile. required
#        only if the filter_cut_off is set
#    mask : array-like (bool)N by M or float optional (None)
#        If an array, the array is used as a mask for the profile, must be the 
#        same shape as the profile, if a float is given, values which match are
#        excluded from the calculation 
#    four_nearest : bool optional (False)
#        If true any point that is higher than it's 4 nearest neigbours will be 
#        counted as a summit, otherwise a point must be higher than it's 8 
#        nearest neigbours to be a summit
#    filter_cut_off : float optional (None)
#        If given the surface will be low pass filtered before finding sumits
#    
#    Returns
#    -------
#    summits : N by M bool array
#        True at location of sumits
#    
#    See Also
#    --------
#    #TODO
#    
#    Notes
#    -----
#    #TODO
#    
#    Examples
#    --------
#    #TODO
#    """
#    if profile is Surface or issubclass(type(profile), Surface):
#        if not grid_spacing:
#            try:
#                grid_spacing=profile._grid_spacing
#            except AttributeError:
#                pass
#        profile=profile.profile
#    
#    profile=np.asarray(profile) # to make sure it's not overwritten by masking
#    
#    if mask is not None:
#        if type(mask) is float:
#            if np.isnan(mask):
#                mask=~np.isnan(profile)
#            else:
#                mask=~profile==mask
#        else:
#            mask=np.asarray(mask, dtype=bool)
#            if not mask.shape==profile.shape:
#                msg=("profile and mask shapes do not match: profile is"
#                    "{profile.shape}, mask is {mask.shape}".format(**locals()))
#                raise TypeError(msg)
#                
#        profile[mask]=float('nan')
#    
#    if filter_cut_off is not None:
#        filtered_profile=low_pass_filter(profile, filter_cut_off, grid_spacing)
#    else:
#        filtered_profile=profile
#    summits=np.ones(profile[1:-1,1:-1].shape, dtype=bool)
#    if four_nearest:
#        x=[-1,+1,0,0]
#        y=[0,0,-1,+1]
#    else:
#        x=[-1,+1,0,0,-1,-1,+1,+1]
#        y=[0,0,-1,+1,-1,+1,-1,+1]
#    
#    for i in range(len(x)):   
#        summits=np.logical_and(summits,(filtered_profile[1:-1,1:-1]>
#                                        filtered_profile[1+x[i]:-1+x[i] or 
#                                                         None,1+y[i]:-1+y[i] 
#                                                         or None]))
#    
#    #pad summits with Falses to make same size as original
#    summits=np.pad(summits, 1, 'constant', constant_values=False)
#    return summits
#    
#def low_pass_filter(profile, cut_off_freq, grid_spacing=None):
#    """2d low pass FIR filter with specified cut off frequency
#    
#    Parameters
#    ----------
#    profile : N by M array-like or Surface object
#        The Surface object or profile to be filtered
#    cut_off_frequency : Float
#        The cut off frequency of the filter in the same units as the 
#        grid_spacing of the profile
#    grid_spacing : float optional (None)
#        The distance between adjacent points of the grid of the surface profile
#        not required if the grid spacing of the Surface object is set, always
#        required when an arraylike profile is used
#        
#    Returns
#    -------
#    filtered_profile : N by M array
#        The filtered surface profile
#    
#    See Also
#    --------
#    Surface
#    
#    Notes
#    -----
#    #TODO
#    
#    Examples
#    --------
#    #TODO
#    
#    References
#    ----------
#    #TODO
#    """
#    if profile is Surface or issubclass(type(profile), Surface):
#        if grid_spacing is None:
#            try:
#                grid_spacing=profile._grid_spacing
#            except AttributeError:
#                pass
#        profile=profile.profile    
#    
#    if grid_spacing is None:
#        msg="Grid spacing must be set"
#        raise ValueError(msg)
#    
#    sz=profile.shape
#    x=np.arange(1, sz[0]+1)
#    y=np.arange(1, sz[1]+1)
#    X,Y=np.meshgrid(x,y)
#    D=np.sqrt(X**2+Y**2)
#    ws=2*np.pi/grid_spacing
#    wc=cut_off_freq*2*np.pi
#    h=(wc/ws)*scipy.special.j1(2*np.pi*(wc/ws)*D)/D
#    filtered_profile=scipy.signal.convolve2d(profile,h,'same')
#    
#    return filtered_profile
#
#def subtract_polynomial(profile, order=1, mask=None): #Checked
#    """ Flattens the surface by fitting and subtracting a polynomial
#    
#    Fits a polynomial to the surface the subtracts it from the surface, to
#    remove slope or curve from imaging machines
#    
#    Parameters
#    ----------
#    
#    order : int
#        The order of the polynomial to be fitted
#    profile : array-like or Surface
#        The surface or profile to be used
#    mask : array-like (bool) same shape as profile or float (defaults to None)
#        If an array, the array is used as a mask for the profile, must be the 
#        same shape as the profile, if a float or list of floats is given, 
#        those values are excluded from the calculation 
#        
#    Returns
#    -------
#    adjusted : array
#        The flattened profile
#    coefs : array
#        The coeficients of the polynomial
#        
#    Examples
#    --------
#    >>>flat_profile, coefs=subtract_polynomial(2, my_surface)
#    Subtract a quadratic polynomial from the profile of my_surface the result
#    is returned but the profile property of the surface is not updated
#    
#    >>>flat_profile, coefs=subtract_polynomial(2,my_surface.profile)
#    Identical to the above opertion
#    
#    >>>flat_profile_2, coefs=subtract_polynomial(1, profile_2)
#    Subtract a plane of best fit from profile_2 and return the result
#    
#    >>>flat_profile, coefs=subtract_polynomial(1, profile_2, mask=float('nan'))
#    Subtract the profile from the surface ignoring nan height values
#    
#    >>>mask=numpy.zeros_like(profile, dtype=bool)
#    >>>mask[5:-5,5:-5]=True
#    >>>flat_profile, coefs=subtract_polynomial(1, profile_2, mask=mask)
#    Subtract a polynomial from the surface ignoring a 5 deep boarder
#    
#    See Also
#    --------
#    roughness
#    numpy.linalg.lstsq
#    
#    Notes
#    -----
#    In principal polynomials of any integer order are supported however higher
#    order polynomials will take more time to fit
#    
#    """
#    if profile is Surface or issubclass(type(profile), Surface):
#        profile=profile.profile
#    
#    profile=np.asarray(profile)
#    x=np.arange(profile.shape[1],dtype=float)
#    y=np.arange(profile.shape[0],dtype=float)
#    Xf,Yf=np.meshgrid(x,y)
#    Zf=profile
#    
#    if mask is not None:
#        if type(mask) is float:
#            if np.isnan(mask):
#                mask=~np.isnan(profile)
#            else:
#                mask=~profile==mask
#        else:
#            mask=np.asarray(mask, dtype=bool)
#            if not mask.shape==profile.shape:
#                msg=("profile and mask shapes do not match: profile is"
#                    "{profile.shape}, mask is {mask.shape}".format(**locals()))
#                raise TypeError(msg)
#        Z=Zf[mask]
#        X=Xf[mask]
#        Y=Yf[mask]
#        
#    else:
#        Z=Zf.flatten()
#        X=Xf.flatten()
#        Y=Yf.flatten()
#    
#    
#    #fit polynomial
#    n_cols=(order+1)**2
#    G=np.zeros((Z.size, n_cols))
#    ij=itertools.product(range(order+1), range(order+1))
#    
#    for k, (i,j) in enumerate(ij):
#        G[:,k]=X**i*Y**j
#        
#    try:
#        coefs, _, _, _ = np.linalg.lstsq(G, Z, rcond=None)
#        
#    except np.linalg.LinAlgError:
#        if any(np.isnan(Z)) or any(np.isinf(Z)):
#            msg="Nans or infs found in surface these should be masked see docs"
#            raise ValueError(msg)
#        else:
#            raise   
#    
#    poly=np.zeros_like(profile)
#    #must reset to itterate again
#    ij=itertools.product(range(order+1), range(order+1))
#    
#    for a, (i,j) in zip(coefs,ij):
#        poly+=a*Xf**i*Yf**j
#    poly=poly.reshape(profile.shape)
#    adjusted=profile-poly
#    
#    if mask is not None:
#        adjusted[~mask]=profile[~mask]
#    
#    return adjusted, coefs

#if __name__=='__main__':
#    import numpy.testing as npt
#    
#    a=np.arange(10)
#    b=np.arange(11)
#    A,B=np.meshgrid(a,b)
#    
#    c=1
#    ac=0.2
#    bc=0.3
#    abc=1.4
#    
#    profile=c+ac*A+bc*B+abc*A*B
#    
#    P,C=subtract_polynomial(profile,1)
#    
#    npt.assert_allclose(C,[c,bc,ac,abc])
#    
#    profile[0,0]=float('nan')
#    
#    assert_raises(ValueError,subtract_polynomial,profile,1)
#    
#    P,C=subtract_polynomial(profile,1,mask=float('nan'))
#    
#    npt.assert_allclose(C,[c,bc,ac,abc])
#    
#    profile[0,0]=1
#    
#    mask=np.logical_or(profile>100, profile<5)
#    
#    P,C=subtract_polynomial(profile,1,mask=mask)
#    
#    npt.assert_allclose(C,[c,bc,ac,abc])
#    
    
#"""
#nan filling
#"""
#import slippy as s
#import numpy as np
#from skimage.restoration import inpaint
#from matplotlib import pyplot
#import warnings
#
#hole_value='auto'
#prop_good=0.99 #0 is none good and 1 is all good
#remove_boarder=True
#
#ms=s.Surface(file_name="D:\\Downloads\\Alicona_data\\Surface Profile Data\\dem.al3d", file_type='.al3d')
#
#profile=ms.profile
#
#if hole_value=='auto':
#    holes=np.logical_or(np.isnan(profile), np.isinf(profile))
#else:
#    holes=profile==hole_value
#    if all(~holes):
#        warnings.warn('No holes detected')
#
#profile[holes]=0
#
#if remove_boarder:
#    # find rows
#    good=[False]*4
#    
#    start_r=0
#    end_r=len(profile)-1
#    start_c=0
#    end_c=len(profile[0])-1
#    
#    while not(all(good)):
#        #start row
#        if 1-sum(holes[start_r,start_c:end_c])/(end_c-start_c)<prop_good:
#            start_r+=1
#        else:
#            good[0]=True
#        
#        #end row
#        if 1-sum(holes[end_r,start_c:end_c])/(end_c-start_c)<prop_good:
#            end_r-=1
#        else:
#            good[1]=True
#    
#        if 1-sum(holes[start_r:end_r,start_c])/(end_r-start_r)<prop_good:
#            start_c+=1
#        else:
#            good[2]=True
#        
#        if 1-sum(holes[start_r:end_r,end_c])/(end_r-start_r)<prop_good:
#            end_c-=1
#        else:
#            good[3]=True
##       
##    
#    profile=profile[start_r:end_r, start_c:end_c]
#    holes=holes[start_r:end_r, start_c:end_c]
### remove all full starting or trailing rows/ cols
#
#        
#
#
#
#image_result = inpaint.inpaint_biharmonic(profile, holes,
#                                          multichannel=False)
#
#pyplot.imshow(profile)




## -*- coding: utf-8 -*-
#"""
#Created on Thu Nov  1 16:37:08 2018
#
#@author: mike
#"""
## read alicona data in 
#
#def alicona_read(full_path):
#    
#    import numpy as np
#    import os
#    import sys#
#    from matplotlib.pyplot import imread
#    
#    path=os.path.split(full_path)[0]
#    
#    data = dict()
#    tags = dict()
#    
#    with open(full_path, 'rb') as file:
#        
#        ###read the header
#        
#        line = file.readline()
#        tags['Type'] = line[:line.find(0)].decode(sys.stdout.encoding)
#        
#        line = file.readline()
#        tags['Version'] = int(bytearray([byte for byte in line[20:-1] if byte!=0]
#                                        ).decode(sys.stdout.encoding))
#        
#        line = file.readline()
#        tags['TagCount'] = int(bytearray([byte for byte in line[20:-1] if byte!=0]
#                                        ).decode(sys.stdout.encoding))
#        
#        for tag_num in range(tags['TagCount']):
#            line=file.readline()
#            tag_name=bytearray([byte for byte in line[0:20] if byte!=0]
#                                ).decode(sys.stdout.encoding)
#            tv_str=bytearray([byte for byte in line[20:-1] if byte!=0]
#                                        ).decode(sys.stdout.encoding)
#            try:
#                tag_value=int(tv_str)
#            except ValueError:
#                try:
#                    tag_value=float(tv_str)
#                except ValueError:
#                    tag_value=tv_str
#            tags[tag_name]=tag_value
#        
#        line=file.readline()
#        tags['comment']=bytearray([byte for byte in line[20:-1] if byte!=0]
#                                  ).decode(sys.stdout.encoding)
#        
#        data['header']=tags
#        
#        #read the icon data
#        
#        if tags['IconOffset']>0:
#            file.seek(tags['IconOffset'])
#            icon=np.zeros([152,150,3], dtype='uint8')
#            for i in range(3):
#                icon[:,:,i]=np.reshape(np.array(file.read(22800), dtype='uint8'), (152,150))
#            data['icon']=icon
#        else:
#            try:
#                icon=imread(path+os.path.sep+"icon.bmp")
#                data['icon']=icon
#            except FileNotFoundError:
#                pass
#        
#        ## read the depth data
#        rows = tags['Rows']
#        
#        if tags['DepthImageOffset']>0:
#            
#            if tags['TextureImageOffset']==0:
#                cols=(file.seek(0,2) - tags['DepthImageOffset'])/(4*rows)
#            else:
#                cols=(tags['TextureImageOffset']-- tags['DepthImageOffset'])/(4*rows)
#            
#            cols= int(round(cols))
#            
#            file.seek(tags['DepthImageOffset'])
#            
#            depth_data=np.array(np.frombuffer(file.read(rows*cols*4), np.float32)) # a single is 4 bytes #TIL # array to make not read only
#            depth_data[depth_data==tags['InvalidPixelValue']]=float('nan')
#            data['DepthData']=np.reshape(depth_data, (rows,cols))[:,:tags['Cols']]
#            
#        #read the texture data
#        
#        if tags['TextureImageOffset']>0:
#            
#            if 'TexturePtr' in tags:
#                if tags['TexturePtr'] == '0;1;2':
#                    num_planes=4
#                else:
#                    num_planes=1    
#            elif 'NumberOfPlanes' in tags:
#                num_planes=tags['NumberOfPlanes']
#            else:
#                msg=("The file format may have been updated please ensure this veri"
#                     "son is up to date then contact the developers")
#                raise NotImplementedError(msg)
#            
#            cols=(file.seek(0,2) - tags['TextureImageOffset'])/(num_planes*rows)
#            
#            file.seek(tags['TextureImageOffset'])
#            
#            texture_data=np.zeros([cols,rows,num_planes], dtype='uint8')
#            
#            for plane in range(num_planes):
#                texture_data[:,:,plane] = np.reshape(np.array(file.read(cols*rows))
#                                                        , (cols,rows))
#            
#            texture_data=texture_data[:tags['Cols'],:,:]
#            
#            if num_planes==4:
#                data['TextureData'] = texture_data[:,:,0:3]
#                data['QualityMap'] = texture_data[:,:,-1]
#            else:
#                data['TextureData'] = texture_data[:,:,0]
#                
#        else:
#            #check if there is a texture image in the current dir
#            try:
#                data['TextureData']=imread(path+os.path.sep+"texture.bmp")
#            except FileNotFoundError:
#                pass
#    return data
#        
## johnson fit by quantiles
##import scipy.stats
##from _johnson_utils import _johnsonsl
##from scipy.stats._continuous_distns import _norm_cdf
##import numpy as np
#
##def _fit_johnson_by_quantiles(quantiles):
##    """
##    Fits a johnson family distribution based on the supplied quartiles
##    
##    Parameters
##    ----------
##    
##    quartiles : array like 4 elements
##        The quartiles to be fitted to 
##    
##    Returns
##    -------
##    
##    dist : scipy.stats._distn_infrastructure.rv_frozen
##    A scipy rv object of the fitted distribution
##    
##    See Also
##    --------
##    
##    References
##    ----------
##    
##    Examples
##    --------
##    
##    """
##    
##    m=quantiles[3]-quantiles[2]
##    n=quantiles[1]-quantiles[0]
##    p=quantiles[2]-quantiles[1]
##    q0=(quantiles[1]+quantiles[2])*0.5
##    
##    mp=m/p
##    nop=n/p
##    
##    tol=1e-4
##    
##    if mp*nop<1-tol:
##        #bounded
##        pm=p/m
##        pn=p/n
##        delta=0.5/np.arccosh(0.5*np.sqrt((1+pm)*(1+pn)))
##        gamma=delta*np.arcsinh((pn-pm)*np.sqrt((1+pm)*(1+pn)-4)/(2*(pm*pn-1)))
##        xlam=p*np.sqrt(((1+pm)*(1+pn)-2)**2-4)/(pm*pn-1)
##        xi=q0-0.5*xlam+p*(pn-pm)/(2*(pm*pn-1))
##        dist=scipy.stats.johnsonsb(gamma, delta, loc=xi, scale=xlam)
##        
##    elif mp*nop>1+tol:
##        #unbounded
##        delta=1/np.arccosh(0.5*(mp+nop))
##        gamma=delta*np.arcsinh((nop-mp)/(2*np.sqrt(mp*nop-1)))
##        xlam=2*p*np.sqrt(mp*nop-1)/((mp+nop-2)*np.sqrt(mp+nop+2))
##        xi=q0+p*(nop-mp)/(2*(mp+nop-2))
##        dist=scipy.stats.johnsonsu(gamma, delta, loc=xi, scale=xlam)
##        
##    elif abs(mp-1)>tol:
##        #lognormal
##        delta=1/np.log(mp)
##        gamma=delta*np.log(abs(mp-1)/(p*np.sqrt(mp)))
##        xlam=np.sign(mp-1)
##        xi=q0-0.5*p*(mp+1)/(mp-1)
##        dist=_johnsonsl(gamma, delta, loc=xi, scale=xlam)
##        
##    else:
##        #normal
##        scale=1/m
##        loc=q0*scale
##        dist=scipy.stats.norm(loc=loc, scale=scale)
##        
##    return dist
##
##if __name__=='__main__':
##    quantile_pts=np.array(_norm_cdf([-1.5,0.5,0.5,1.5]))
##    test_quantiles=[[1,1.2,1.4,10]]
##    
##    for quantiles_in in test_quantiles:
##        dist=_fit_johnson_by_quantiles(quantiles_in)
##        quantiles_out=dist.ppf(quantile_pts)
#        
#
#
### filter coeficents kurtosis transfromation 
##
##import numpy as np
##from numpy.matlib import repmat
##
##alpha=np.arange(5)
##alpha2=alpha**2
##alphapad2=np.pad(alpha2, [0, len(alpha2)], 'constant')
##ai=repmat(alpha2[:-1], len(alpha2)-1,1)
##
##idX,idY=np.meshgrid(np.arange(len(alpha2)-1),np.arange(len(alpha2)-1))
##index=idX+idY+1 # diagonally increaing matrix
##
##aj=alphapad2[index]
##
##quad_term=sum(ai*aj)
#
### Johnson translator system
#
##def fit_johnson_by_moments(mean, sd, root_beta_1, beta_2):
##    """
##    Fits a johnson family distribution to the specified moments
##    
##    Parameters
##    ----------
##    mean : scalar, The population mean
##    sd : scalar, The population standard deviation
##    root_beta_1 : scalar, the skew of the distribution
##    beta_2 : scalar, the kurtosis of the distribution (not normalised)
##    
##    Returns
##    -------
##    
##    DistType : {1 - lognormal johnson Sl, 2 - unbounded johnson Su, 3 bounded 
##                johnson Sb, 4 - normal, 5 - Boundry johnson St} 
##        integer, a number corresponding to the type of distribution that has
##        been fitted
##    xi, xlam, gamma, delta : scalar, shape parameters of the fitted 
##        distribution, xi is epsilon, xlam is lambda
##    
##    When a normal distribution is fitted (type 4) the delta is set to 1/sd and 
##    gamma is set to mean/sigma, xi and xlam are arbitrarily set to 0.
##    
##    When a boundry johnson curve is fitted (type 5, St) the return parameters 
##    have different meanings. xi and xlam are set to the two values at which 
##    ordinates occur and delta to the proportion of values at xlam, gamma is 
##    set arbitrarily to 0.
##    
##    See Also
##    -------
##    ##TODO whatever function i wrap this up with that returns a scipy distribution, i'm over running this line so it looks stupid and you'll try to fix it them realise 
##    
##    Notes
##    -----
##    Coppied from algorithm 99.3 in
##    applied statistics, 1976 vol 25 no 2 
##    accesable here:
##    https://www.jstor.org/stable/pdf/2346692.pdf
##    also in c as part of the R supdists package here:
##    https://cran.r-project.org/web/packages/SuppDists/index.html
##    
##    changes from the original functionallity are noted by #CHANGE
##    
##    Examples
##    --------
##    
##    >>>make me a sandwich ##TODO 
##    ##TODO
##    there is some other logic that is not implemented, go back the the main program, somthing happens whene there is a fault with the sb fitting that should happen here too
##    """
##    
##    import math
##    tollerance=0.01
##    
##    beta_1=root_beta_1**2
##    
##    xi=0
##    xlam=0
##    gamma=0
##    delta=0
##    dist_type=0
##    
##    if sd<0:
##        raise ValueError("Standard deviation must be grater than or equal to 0") # error code 1
##    elif sd==0:
##        xi=mean 
##        dist_type=5 #ST distribution
##        return (dist_type, xi, xlam, gamma, delta)
##    
##    if beta_2>=0:
##        if beta_2<beta_1+1-tollerance:
##            raise ValueError("beta 2 must be greater than or eqaul to beta 1 + 1")# error code 2 
##        if beta_2<=(beta_1+tollerance+1):
##            dist_type=5 #ST distribution
##            y=0.5+0.5*math.sqrt(1-4/(beta_1+4))
##            if root_beta_1>0:
##                y=1-y
##            x=sd/math.sqrt(y*(1-y))
##            xi=mean-y*x
##            xlam=xi+x
##            delta=y
##            return (dist_type, xi, xlam, gamma, delta)
##    
##    if abs(root_beta_1)<tollerance and abs(beta_2-3)<tollerance:
##        dist_type=4 #Normal
##        xi=mean
##        xlam=sd
##        delta=1.0
##        gamma=0.0 #CHANGE from hill, in line with R package
##        return (dist_type, xi, xlam, gamma, delta)
##    
##    #80
##    # find critical beta_2 (lies on lognormal line)
##    x=0.5*beta_1+1
##    y=root_beta_1*math.sqrt(0.25*beta_1+1)
##    omega=(x+y)**(1/3)+(x-y)**(1/3)-1
##    beta_2_lognormal=omega*omega*(3+omega*(2+omega))-3
##    
##    if beta_2<0 or False:#there is a fault var here but i can't figure out whatit's doing must be global set in sub function returns lognormal solution if sb fit fails 
##        beta_2=beta_2_lognormal
##    
##    # if x is 0 log normal, positive - bounded, negative - unbounded
##    x=beta_2_lognormal-beta_2
##    
##    if abs(x)<tollerance:
##        dist_type=1 #log normal
##        if root_beta_1<0:
##            xlam=-1
##        else:
##            xlam=1
##        u=xlam*mean
##        x=1/math.sqrt(math.log(omega))
##        delta=x
##        y=0.5*x*math.log(omega*(omega-1)/(sd**2))
##        gamma=y
##        xi=u-math.exp((0.5/x-y)/x)
##        return (dist_type, xi, xlam, gamma, delta)
##    
##    
##    def get_moments(g, d, max_it=500, tol_outer=1e-5, tol_inner=1e-8):
##        """
##        Evaluates the first 6 moments of a johnson distribution using goodwin's
##        method (SB only)
##         
##        Parameters
##        ----------
##            g : scalar, shape parameter
##            d : scalar, shape parameter
##        
##        Returns
##        -------
##        moments : list of the first 6 moments
##        
##        Notes
##        -----
##        Coppied from algorithm 99.3 in
##        applied statistics, 1976 vol 25 no 2 
##        accesable here:
##            https://www.jstor.org/stable/pdf/2346692.pdf
##            
##        See also
##        --------
##        sb_fit : fits bounded johnson distribuitions 
##        fit_johnson_by_moments : fits johnson family distributions by moments
##        """
##        
##        
##        moments=6*[1]
##        b=6*[0]
##        c=6*[0]
##        
##        w=g/d
##        
##        # trial value of h
##        
##        if w>80:
##            raise ValueError("Some value too high, failed to converge")
##        e=math.exp(w)+1
##        r=math.sqrt(2)/d
##        h=0.75
##        if d<3:
##            h=0.25*d
##        k=0
##        h*=2
##        while any([abs(A-C)/A>tol_outer for A, C in zip(moments, c)]):
##            k+=1
##            if k>max_it:
##                raise StopIteration("Moment filnding failed to converge O")
##            if k>1:
##                c=list(moments)
##            
##            # no convergence yet try smaller h
##            
##            h*=0.5
##            t=w
##            u=t
##            y=h**2
##            x=2*y
##            moments[0]=1/e
##            
##            for i in range(1,6):
##                moments[i]=moments[i-1]/e
##            
##            v=y
##            f=r*h
##            m=0
##            
##            # inner loop to evaluate infinite series
##            while any([abs(A-B)/A>tol_inner for A, B in zip(moments, b)]):
##                m+=1
##                if m>max_it:
##                    raise StopIteration("Moment filnding failed to converge I")
##                b=list(moments)
##                u=u-f
##                z=math.exp(u)+1
##                t=t+f
##                #######change made here, not any more just printing
##                l=t>23.7
##                if not l:
##                    s=math.exp(t)+1
##                p=math.exp(-v)
##                q=p
##                for i in range(1,7):
##                    moment=moments[i-1]
##                    moment_a=moment
##                    
##                    p=p/z
##                    if p==0:
##                        break
##                    moment_a=moment_a+p
##                    if not l:
##                        q=q/s
##                        moment_a=moment_a+q
##                        l=q==0
##                    moments[i-1]=moment_a
##                #100
##                y=y+x
##                v=v+y
##                
##                if any([moment==0 for moment in moments]):
##                    raise ValueError("for some reason having zero moments"
##                                     " is not allowed, you naughty boy")
##                    #######carry on from here
##            #end of inner loop
##            v=1/math.sqrt(math.pi)*h
##            moments=[moment*v for moment in moments]
##            #don't need to check all non zero here, just checked!
##        #end of outer loop
##        return moments
##    
##    def sb_fit(mean, sd, root_beta_1, beta_2, tollerance, max_it=50):
##        dist_type=3
##        
##        beta_1=root_beta_1**2
##        neg=root_beta_1<0
##        
##        # get d as first estimate of delta
##        
##        e=beta_1+1
##        u=1/3
##        x=0.5*beta_1+1
##        y=abs(root_beta_1)*math.sqrt(0.25*beta_1+1)
##        w=(x+y)**u+(x-y)**u-1
##        f=w**2*(3+w*(2+w))-3
##        e=(beta_2-e)/(f-e)
##        if abs(root_beta_1)<tollerance:
##            f=2
##        else:
##            d=1/math.sqrt(math.log(w))
##            if d>=0.04:
##                f=2-8.5245/(d*(d*(d-2.163)+11.346))
##            else:
##                f=1.25*d
##        f=e*f+1 #20
##        
##        if f<1.8:
##            d=0.8*(f-1)
##        else:
##            d=(0.626*f-0.408)*(3-f)**(-0.479)
##        
##        # get g as a first estimate of gamma
##        
##        g=0 #30
##        if beta_1>tollerance**2:
##            if d<=1:
##                g=(0.7466*d**1.7973+0.5955)*beta_1**0.485
##            else:
##                if d<=2.5:
##                    u=0.0623
##                    y=0.5291
##                else:
##                    u=0.0124
##                    y=0.5291
##                g=beta_1**(u*d+y)*(0.9281+d*(1.0614*d-0.7077))
##        
##        # main itterations start here
##        m=0
##        
##        u=float('inf')
##        y=float('inf')
##        
##        dd=[0]*4
##        deriv=[0]*4
##        
##        while abs(u)>tollerance**2 or abs (y)>tollerance**2:
##            m+=1
##            if m>max_it:
##                import warnings
##                warnings.warn('soultion failed to converge error greater than'
##                              ' the specified tolerance may be present')
##                return False
##            
##            # get first six moments
##            moments=get_moments(g,d)
##            s=moments[0]**2
##            h2=moments[1]-s
##            if h2<=0:
##                raise ValueError("Solution failed to converge")
##                return False
##            t=math.sqrt(h2)
##            h2a=t*h2
##            h2b=h2**2
##            h3=moments[2]-moments[0]*(3*moments[1]-2*s)
##            rbet=h3/h2a
##            h4=moments[3]-moments[0]*(4*moments[2]
##                                        -moments[0]*(6*moments[1]-3*s))
##            bet2=h4/h2b
##            w=g*d
##            u=d**2
##            
##            # get derivatives
##            
##            for j in range(1,3):
##                for k in range(1,5):
##                    t=k
##                    if not j==1:
##                        s=((w-t)*(moments[k-1]-moments[k])+(t+1)*
##                           (moments[k]-moments[k+1]))/u
##                    else:
##                        s=moments[k]-moments[k-1]
##                    dd[k-1]=t*s/d
##                t=2*moments[0]*dd[0]
##                s=moments[0]*dd[1]
##                y=dd[1]-t
##                deriv[j-1]=(dd[2]-3*(s+moments[1]*dd[0]-t*moments[0]
##                            )*-1.5*h3*y/h2)/h2a
##                deriv[j+1]=(dd[3]-4*(dd[2]*moments[0]+dd[0]*moments[2])+6*
##                           (moments[1]*t+moments[0]*(s-t*moments[0]))
##                            -2*h4*y/h2)/h2b
##            
##            t=1/(deriv[0]*deriv[3]-deriv[1]*deriv[2])
##            u=(deriv[3]*(rbet-abs(root_beta_1))-deriv[1]*(bet2-beta_2))*t
##            y=(deriv[0]*(bet2-beta_2)-deriv[2]*(rbet-abs(root_beta_1)))*t
##            
##            # new estimates for G and D
##            
##            g=g-u
##            if beta_1==0 or g<0:
##                g=0
##            d=d-y
##        
##        #end of itteration
##        
##        delta=d
##        xlam=sd/math.sqrt(h2)
##        if neg:
##            gamma=-g
##            moments[0]=1-moments[0]
##        else:
##            gamma=g
##        xi=mean-xlam*moments[0]
##        
##        return (dist_type, xi, xlam, gamma, delta)
##    
##    
##    def su_fit(mean, sd, root_beat_1, beat_2, tollerance):
##        dist_type=2
##        
##        beta_1=root_beta_1**2
##        b3=beta_2-3
##        
##        #first estimate of e**(delta**(-2))
##        
##        w=math.sqrt(math.sqrt(2.0*beta_2-2.8*beta_1-2.0)-1.0)
##        if abs(root_beta_1)<tollerance:
##            #symetrical case results known
##            y=0
##        else:
##            z=float('inf')
##            #johnson itterations
##            while abs(beta_1-z)>tollerance:
##                w1=w+1
##                wm1=w-1
##                z=w1*b3
##                v=w*(6+w*(3+w))
##                a=8*(wm1*(3+w*(7+v))-z)
##                b=16*(wm1*(6+v)-b3)
##                m=(math.sqrt(a**2-2*b*(wm1*(3+w*(9+w*(10+v)))-2*w1*z))-a)/b
##                z=m*wm1*(4*(w+2)*m+3*w1**2)**2/(2*(2*m+w1)**3)
##                v=w**2
##                w=math.sqrt(math.sqrt(1-2*(1.5-beta_2+(beta_1*(
##                              beta_2-1.5-v*(1+0.5*v)))/z))-1)
##            y=math.log(math.sqrt(m/w)+math.sqrt(m/w+1))
##            if root_beta_1>0:
##                y*=-1
##        x=math.sqrt(1/math.log(w))
##        delta=x
##        gamma=y*x
##        y=math.exp(y)
##        z=y**2
##        x=sd/math.sqrt(0.5*(w-1)*(0.5*w*(z+1/z)+1))
##        xlam=x
##        xi=(0.5*math.sqrt(w)*(y-1/y))*x*mean
##        
##        return (dist_type, xi, xlam, gamma, delta)
##    
##    if x>0:
##        return sb_fit(mean, sd, root_beta_1, beta_2, tollerance)
##    else:
##        return su_fit(mean, sd, root_beta_1, beta_2, tollerance)
##
##def johnson_fit_wrapper(mean, sd, root_beta_1, beta_2, stats=False):
##    import scipy.stats
##    dist_type, xi, xlam, gamma, delta=fit_johnson_by_moments(mean, sd, root_beta_1, beta_2)
##    #need to figure out how to multiply and add to this... maybe provide a wrapper function?
##    if stats:
##        if dist_type==1: # sl
##            ###? might need to implement can't find in 
##            #TODO
##            print('sL\n')
##            pass
##        elif dist_type==2: #su
##            print('su\n')
##            stats=scipy.stats.johnsonsu.stats(gamma, delta, loc=xi, scale=xlam, moments='mvsk')
##        elif dist_type==3: #sb
##            print('sb\n')
##            stats=scipy.stats.johnsonsb.stats(gamma, delta, loc=xi, scale=xlam, moments='mvsk')
##        elif dist_type==4: #st
##            print('normal\n')
##            stats=scipy.stats.norm.stats(loc=xi, scale=xlam, moments='mvsk')
##        elif dist_type==5: # normalST
##            print('st\n')
##            # st distributions are one or 2 values, not very useful for surface analysis
##            raise NotImplementedError("ST distributions are not implemented")
##            #might need to implement this too 
##        #print(stats)
##        return stats
##    
##    if dist_type==1: # sl
##        ###? might need to implement can't find in 
##        #TODO
##        pass
##    elif dist_type==2: #su
##        dist=scipy.stats.johnsonsu(gamma, delta, loc=xi, scale=xlam)
##    elif dist_type==3: #sb
##        dist=scipy.stats.johnsonsb(gamma, delta, loc=xi, scale=xlam)
##    elif dist_type==4: #st
##        # st distributions are one or 2 values, not very useful for surface analysis
##        raise NotImplementedError("ST distributions are not implemented")
##        #might need to implement this too 
##    elif dist_type==5: # normal
##        dist=scipy.stats.normal(loc=xi, scale=xlam)
##    return dist
##    
##if __name__=='__main__':
##    from math import sqrt
##    
##    #normal
##    #dist_type, xi, xlam, gamma, delta=fit_johnson_by_moments(5,2,0,3)
##    result=johnson_fit_wrapper(5,2,0,3,True)
##    #print(result)
##    # should return [4, 0, 0, 2.5, 0.5]
##    normal=fit_johnson_by_moments(0,1,0,3)
##    bounded=fit_johnson_by_moments(5,2,sqrt(2),4)
##    unbounded=fit_johnson_by_moments(5,2,1,6)
##    boarderline=fit_johnson_by_moments(5,2,sqrt(2),3)
#    #impossible=fit_johnson_by_moments(5,2,sqrt(4),2)
#    
#
################################################################################
################################################################################
########### netonian itterations for a random surface
################################################################################
################################################################################
#
##
##import numpy as np
##max_it=100
##accuracy=0.0001
##min_relax=1e-6
### N by M surface
##
### n by m ACF
##n=51
##m=51
##spacing=1
##
##l=np.arange(n)
##k=np.arange(m)
##[K,L]=np.meshgrid(k,l)
##
##sigma=0.5
##beta_x=15/spacing
##beta_y=15/spacing
##ACF=sigma**2*np.exp(-2.3*np.sqrt((K/beta_x)**2+(L/beta_y)**2))
##
#### initial guess (n by m guess of filter coefficents)
##c=ACF/((m-K)*(n-L))
##s=np.sqrt(ACF[0,0]/np.sum(c.flatten()**2))
##
##alpha=s*c
##it_num=0
##As=[]
##relaxation_factor=1
##
##f=np.zeros_like(alpha)
##for p in range(n):
##    for q in range(m):
##       f[p,q]=np.sum(alpha[0:n-p,0:m-q]*alpha[p:n,q:m])
##f=(f-ACF).flatten()
##
##resid_old=np.sqrt(np.sum(f**2))
##
##print('Itteration started:\nNumber\tResidual\t\tRelaxation factor')
##
##while it_num<max_it and relaxation_factor>min_relax and resid_old>accuracy:
##    
##    #make Jackobian matrix
##    A0=np.pad(alpha, ((n,n),(m,m)), 'constant')
##    Jp=[]
##    Jm=[]
##    
##    for p in range(n):
##        Jp.extend([A0[n+p:2*n+p,m+q:2*m+q].flatten() for q in range(m)])
##        Jm.extend([A0[n-p:2*n-p,m-q:2*m-q].flatten() for q in range(m)])
##    
##    J=(np.array(Jp)+np.array(Jm))
##    
##    #doing thw itteration
##    change=np.matmul(np.linalg.inv(J),f)
##    
##    resid=float('inf')
##    while resid>=resid_old:    
##        alpha_new=(alpha.flatten()-relaxation_factor*change).reshape((n,m))
##        
##        #make f array to find residuals
##        
##        f_new=np.zeros_like(alpha)
##        for p in range(n):
##            for q in range(m):
##                f_new[p,q]=np.sum(alpha_new[0:n-p,0:m-q]*alpha_new[p:n,q:m])
##        f_new=(f_new-ACF).flatten()
##        #print the residuals
##        resid=np.sqrt(np.sum(f_new**2))
##        relaxation_factor/=2
##    
##    print('{it_num}\t{resid}\t{relaxation_factor}'.format(**locals()))
##    
##    relaxation_factor=min(relaxation_factor*8,1)
##    alpha=alpha_new
##    f=f_new
##    resid_old=resid
##    it_num+=1
##
##if resid<accuracy:
##    print('Itteration stopped, sufficent accuracy reached')
##elif relaxation_factor<min_relax:
##    print('Itteration stoped, local minima found, residual is {resid}'.format(**locals()))
##elif it_num==max_it:
##    print('Itteration stopped, no convergence after {it_num} itterations, residual is {resid}'.format(**locals())) 
##    
######### end filter generation start filtering and surface generation 
###N by M surface
##N=512
##M=512
##periodic=False
##
##import warnings
##
##if periodic:
##    if n%2==0 or m%2==0:
##        msg='For a periodic surface the filter coeficents matrix must have an odd number of elements in every dimention, output profile will not be the expected size'
##        warnings.warn(msg)
##    pad_rows=int(np.floor(n/2))
##    pad_cols=int(np.floor(m/2))
##    eta=np.pad(np.random.randn(N, M),((pad_rows,pad_rows),(pad_cols,pad_cols)),'wrap')
##else: 
##    eta=np.random.randn(N+n-1, M+m-1)
##
##import scipy.signal
##
##profile=scipy.signal.fftconvolve(eta, alpha, 'valid')
#
#
################################################################################
################################################################################
########### hurst fractal
################################################################################
################################################################################
#
##import numpy as np
##
##q_cut_off=100
##q0=1
##q0_amp=1
##Hurst=2
##N=int(round(q_cut_off/q0))
##h, k=range(-1*N,N+1), range(-1*N,N+1)
##H,K=np.meshgrid(h,k)
###generate values
##mm2=q0_amp**2*((H**2+K**2)/2)**(1-Hurst)
##mm2[N,N]=0
##pha=2*np.pi*np.random.rand(mm2.shape[0], mm2.shape[1])
##
##mean_mags2=np.zeros((2*N+1,2*N+1))
##phases=np.zeros_like(mean_mags2)
##
##mean_mags2[:][N:]=mm2[:][N:]
##mean_mags2[:][0:N+1]=np.flipud(mm2[:][N:])
##mean_mags=np.sqrt(mean_mags2).flatten()
##
##phases[:][N:]=pha[:][N:]
##phases[:][0:N+1]=np.pi*2-np.fliplr(np.flipud(pha[:][N:]))
##phases[N,0:N]=np.pi*2-np.flip(phases[N,N+1:])
##
##phases=phases.flatten()
##
##X,Y=np.meshgrid(range(100),range(100))
##
##coords=np.array([X.flatten(), Y.flatten()])
##
##K=np.transpose(H)
##
##qkh=np.transpose(np.array([q0*H.flatten(), q0*K.flatten()]))
##
##Z=np.zeros(X.size,dtype=np.complex64)
##
##for idx in range(len(qkh)):
##    Z+=mean_mags[idx]*np.exp(1j*(np.dot(qkh[idx],coords)-phases[idx]))
##
##Z=Z.reshape(X.shape)