import numpy as np
import warnings
from math import ceil
import scipy.signal

__all__=['Surface']

class Surface():
    """ the basic surface class contains methods for setting properties, 
    examining measures of roughness and descriptions of surfaces, plotting,
    fixing and editing surfaces.
    
    Adding surfaces together produces a surface with the same properties as the
    original surfaces with a profile that is the sum of the original profiles.
    
    Multiplying 1D surfaces produces a 2D surface with a profile of the 
    summation of surface 1 stretched in the x direction and surface 2 
    stretched in the y direction.
    
    #TODO:
        Then start checking stuff
        def open(work with save)
        make init work with these options 
        Add desciption and example of each method 
        check plotting
        check summit finding
        def save (save whole surface with all info)
        def export #gmsh?
        
        Make this pretty ^
    """
    # The surface class for descrete surfaces (typically experiemntal)
    is_descrete=True
    acf=False
    aacf=False
    psd=False
    fft=False
    hist=False
    profile=np.array([])
    pts_each_direction=[]
    sa=False
    surface_type="Generic"
    dimentions=2
    
    def __init__(self,*args,**kwargs):
        # initialisation surface
        self.init_checks(kwargs)
        # check for file
        if 'file_name' in kwargs:
            ext=kwargs['file_name'][-3:]
            if ext=='pkl':
                self.load_from_file(kwargs['file_name'])
            else:
                try:
                    self.read_from_file(kwargs['file_name'], kwargs['file_type'], **kwargs)
                except KeyError:
                    msg=("file_type keyword should be set to use read from file"
                         " using extention as file type")
                    warnings.warn(msg)
                    self.read_from_file(kwargs['file_name'], ext, **kwargs)
        
    def init_checks(self, kwargs=False):
        # add anything you want to run for all surface types here
        if kwargs:
            allowed_keys = ['global_size', 'grid_size', 'dimentions', 
                            'is_descrete', 'profile', 'pts_each_direction']
            self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
    
    def descretise_checks(self):
        if self.is_descrete:
            msg='Surface is already discrete'
            raise ValueError(msg)
        try: 
            spacing=self.grid_size
        except AttributeError:
            msg='A grid size must be provided before descretisation'
            raise AttributeError(msg)
        try:
            pts_each_direction=[int(gs/spacing) for gs in 
                                    self.global_size]
            total_pts=1
            for pts in pts_each_direction:
                total_pts *= pts
            self.total_pts=total_pts
            self.pts_each_direction=pts_each_direction
        except AttributeError:
            msg='Global size must be set before descretisation'
            raise AttributeError(msg)
        if total_pts>10E7:
            warnings.warn('surface contains over 10^7 points calculations will'
                          ' be slow, consider splitting surface or analysis')
            
    def get_fft(self, surf_in=False):
        if type(surf_in) is bool:
            profile=self.profile
        else:
            profile=surf_in
        try:
            if self.dimentions==1:
                transform=np.fft.fft(profile)
                if type(surf_in) is bool: self.fft=transform
            else:
                transform=np.fft.fft2(profile)
                if type(surf_in) is bool: self.fft=transform
        except AttributeError:
            raise AttributeError('Surface must have a defined profile for fft'
                                 ' to be used')
            
        return transform
    
    def get_acf(self, surf_in=False):
        if type(surf_in) is bool and not surf_in:
            profile=self.profile
        else:
            profile=np.asarray(surf_in)
        x=profile.shape[0]
        y=profile.shape[1]
        output=(scipy.signal.correlate(profile,profile,'same')/(x*y))
        if type(surf_in) is bool:
            self.acf=output
        return output

    def get_psd(self):
        # PSD is the fft of the ACF (https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density)
        if type(self.acf) is bool:
            self.get_acf()
        self.psd=self.get_fft(self.acf)
    
    def subtract_polynomial(self, order, surf_in):
        import itertools
        if not surf_in is bool:
            profile=surf_in
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
        if surf_in is bool:
            self.profile=adjusted
        return adjusted

    def birmingham(self, parameter_name, curved_surface=False, 
                   periodic_surface=False, p=None, **kwargs): #TODO add examples
        """
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
        gs2=self.grid_size**2
        p_area=(self.profile.shape[0]-1)*(
                    self.profile.shape[1]-1)*gs2
        
        if parameter_name=='sq': #root mean square
            out=np.sqrt(np.mean(eta**2))
            
        elif parameter_name=='sa': #mean amptitude
            out=np.mean(np.abs(eta))
            
        elif parameter_name=='ssk': #skewness
            sq=np.sqrt(np.mean(eta**2))
            out=np.mean(eta**3)/sq**3
            
        elif parameter_name=='sku': #kurtosis
            sq=np.sqrt(np.mean(eta**2))
            out=np.mean(eta**4)/sq**4
            
        elif parameter_name in ['sds', 'sz', 'ssc']: # all that require sumits
            # summits is logical array of sumit locations
            kwargs['profile']=eta
            summits=self.find_summits(kwargs)
            if parameter_name=='sds': # summit density
                out=np.sum(summits)/(self.global_size[0]*self.global_size[1])
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
            x=self.grid_size*np.arange(self.profile.shape[0]/-2,
                                       self.profile.shape[0]/2)
            y=self.grid_size*np.arange(self.profile.shape[1]/-2,
                                       self.profile.shape[1]/2)
            X,Y=np.meshgrid(x,y)
            distance_to_centre=np.sqrt(X**2+Y**2)
            min_dist=min(distance_to_centre[self.acf<0.2])-1
            max_dist=max(distance_to_centre[self.acf>0.2])

            out=min_dist/max_dist

        elif parameter_name=='std': # surface texture direction
            if type(self.fft) is bool:
                self.get_fft()
            apsd=self.fft*np.conj(self.fft)/p_area
            x=self.grid_size*np.arange(self.profile.shape[0])
            y=self.grid_size*np.arange(self.profile.shape[1])
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
            x=self.grid_size*np.arange(self.profile.shape[0]/-2,
                                       self.profile.shape[0]/2)
            y=self.grid_size*np.arange(self.profile.shape[1]/-2,
                                       self.profile.shape[1]/2)
            X,Y=np.meshgrid(x,y)
            distance_to_centre=np.sqrt(X**2+Y**2)
            
            out=min(distance_to_centre[self.acf<0.2])-1
        else:
            msg='Paramter name not recognised'
            ValueError(msg)
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
                ValueError('Volume ratio must be bwteen 0 and 1')
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
                )/self.grid_size**2 for vert in verts]
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
                              filtered_profile[1+x[i]:-1+x[i] or None,1+y[i]:-1+y[i] or None]))
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
        ws=2*np.pi/self.grid_size
        wc=cut_off_freq*2*np.pi
        h=(wc/ws)*scipy.special.j1(2*np.pi*(wc/ws)*D)/D
        filtered_profile=scipy.signal.convolve2d(profile,h,'same','wrap')
        if copy:
            return filtered_profile
        else:
            self.profile=filtered_profile
            return
    
    def read_from_file(self, file_name, file_type, *args, **kwargs):
        """ 
        kwargs can be header lines etc, args is kwargs from init
        """
        
        try:
            file_type=file_type.lower()
        except AttributeError:
            msg="file_type should be a string"
            ValueError(msg)
        if file_type=='csv' or file_type=='txt':
            import csv
            if 'delimiter' in kwargs:
                delimiter=kwargs['delimiter']
            else:
                delimiter=','
            with open(file_name) as file:
                if delimiter == ' ' or delimiter == '\t':
                    reader=csv.reader(file, delim_whitespace=True)
                else:
                    reader=csv.reader(file, delimiter=delimiter)
                profile=[]
                for row in reader:
                    if len(row)==1:
                        try:
                            row=[float(x) for x in row[0].split()]
                        except ValueError:
                            pass
                    if len(row)>1:
                        profile.append(row)
            self.profile=np.array(profile)
        
    def fit_johnson_translator(self):
        pass
        #TODO finish this 
    
    def resample(self, new_grid_size, copy=False, kind='cubic'):
        import scipy.interpolate
        x0=np.arange(0, self.global_size[0], self.grid_size)
        y0=np.arange(0, self.global_size[1], self.grid_size)
        X0,Y0=np.meshgrid(x0,y0)
        inter_func=scipy.interpolate.interp2d(X0, Y0, self.profile, kind=kind)
        x1=np.arange(0, self.global_size[0], new_grid_size)
        y1=np.arange(0, self.global_size[1], new_grid_size)
        X1,Y1=np.meshgrid(x1,y1)
        inter_profile=inter_func(X1,Y1)
        if copy:
            return inter_profile
        else:
            self.profile=inter_profile
            self.grid_size=new_grid_size
            self.profile.shape=[len(x1),len(y1)]
        
    def __add__(self, other):
        if self.global_size==other.global_size:
            if self.grid_size==other.grid_size:
                out=Surface(profile=self.profile+other.profile, 
                            global_size=self.global_size, 
                            grid_size=self.grid_size, 
                            is_descrete=self.is_descrete, 
                            pts_each_direction=self.profile.shape, 
                            total_pts=self.total_pts)
                return out
            else:
                msg="Surface sizes do not match: resampling"
                warnings.warn(msg)
                ## resample surface with coarser grid then add again
                if self.grid_size>other.grid_size:
                    self.resample(other.grid_size, False)
                else:
                    other.resample(self.grid_size)
                    
                return self+other
        elif self.profile.shape==other.profile.shape:
            msg=("number of points in surface matches by size of surfaces are"
                "not the same, this operation will add the surfaces point by" 
                "point but this may cause errors")
            warnings.warn(msg)
            out=Surface(profile=self.profile+other.profile, 
                            global_size=self.global_size, 
                            grid_size=self.grid_size, 
                            is_descrete=self.is_descrete, 
                            pts_each_direction=self.profile.shape, 
                            total_pts=self.total_pts)
            return out
        else:
            ValueError('surfaces are not compatible sizes cannot add')
            
    def __sub__(self, other):
        if all(self.global_size==other.global_size):
            if self.grid_size==other.grid_size:
                out=Surface(profile=self.profile-other.profile, 
                            global_size=self.global_size, 
                            grid_size=self.grid_size, 
                            is_descrete=self.is_descrete, 
                            pts_each_direction=self.profile.shape, 
                            total_pts=self.total_pts)
                return out
            else:
                msg="Surface sizes do not match: resampling"
                warnings.warn(msg)
                ## resample surface with coarser grid then add again
                if self.grid_size>other.grid_size:
                    self.resample(other.grid_size, False)
                else:
                    other.resample(self.grid_size)
                    
                return self-other
        elif all(self.profile.shape==other.profile.shape):
            msg=("number of points in surface matches by size of surfaces are"
                "not the same, this operation will subtract the surfaces point"
                "by point but this may cause errors")
            warnings.warn(msg)
            out=Surface(profile=self.profile-other.profile, 
                            global_size=self.global_size, 
                            grid_size=self.grid_size, 
                            is_descrete=self.is_descrete, 
                            pts_each_direction=self.profile.shape, 
                            total_pts=self.total_pts)
            return out
        else:
            ValueError('surfaces are not compatible sizes cannot subtract')
        
    def show(self, property_to_plot='profile', plot_type='default', ax=False, *args):
        """ Method for plotting anything of interest for the surface
        
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
        if not ax:
            fig = plt.figure()
        
        property_to_plot=property_to_plot.lower()
        
        if property_to_plot in types2d:
            if not ax and not plot_type=='image':
                ax=fig.add_subplot(111, projection='3d')
            elif not ax and plot_type=='image':
                ax=fig.add_subplot(111)
                
            if property_to_plot=='profile':
                labels=['Surface profile', 'x', 'y', 'Height']
                Z=self.profile
                x=self.grid_size*np.arange(self.profile.shape[0])
                y=self.grid_size*np.arange(self.profile.shape[1])
                
            elif property_to_plot=='fft2d':
                labels=['Fourier transform of surface', 'u', 'v', '|F(x)|']
                if type(self.fft) is bool:
                    self.get_fft()
                Z=np.abs(np.fft.fftshift(self.fft))
                x=np.fft.fftfreq(self.profile.shape[0], self.grid_size)
                y=np.fft.fftfreq(self.profile.shape[1], self.grid_size)
                
            elif property_to_plot=='psd':
                labels=['Power spectral density', 'u', 'v', 'Power/ frequency']
                if type(self.psd) is bool:
                    self.get_psd()
                Z=np.abs(np.fft.fftshift(self.psd))
                x=np.fft.fftfreq(self.profile.shape[0], self.grid_size)
                y=np.fft.fftfreq(self.profile.shape[1], self.grid_size)
                
            elif property_to_plot=='acf':
                labels=['Auto corelation function', 'x', 'y', 
                        'Surface auto correlation']
                if type(self.acf) is bool:
                    self.get_acf()
                Z=np.abs(self.acf)
                x=self.grid_size*np.arange(self.profile.shape[0])
                y=self.grid_size*np.arange(self.profile.shape[1])
                x=x-max(x)/2
                x=y-max(y)/2
            elif property_to_plot=='apsd':
                labels=['Angular power spectral density', 'x', 'y']
                if type(self.fft) is bool:
                    self.get_fft()
                p_area=(self.profile.shape[0]-1)*(
                    self.profile.shape[1]-1)*self.grid_size**2
                Z=self.fft*np.conj(self.fft)/p_area
                x=self.grid_size*np.arange(self.profile.shape[0])
                y=self.grid_size*np.arange(self.profile.shape[1])
                x=x-max(x)/2
                x=y-max(y)/2 
            
            X,Y=np.meshgrid(x,y)
            
            if plot_type=='default' or plot_type=='surface':
                ax.plot_surface(X,Y,Z)
                ax.set_zlabel(labels[3])
            elif plot_type=='mesh':
                if property_to_plot=='psd' or property_to_plot=='fft2d':
                    X, Y = np.fft.fftshift(X), np.fft.fftshift(Y)
                if args:
                    ax.plot_wireframe(X, Y, Z, rstride=args[0], 
                                      cstride=args[1])
                else:
                    ax.plot_wireframe(X, Y, Z, rstride=25, cstride=25)
                ax.set_zlabel(labels[3])
            elif plot_type=='image':
                ax.imshow(Z, extent=[min(x),max(x),min(y),max(y)], aspect=1)
            #elif plot_type=='image_wrap':
            #    if property_to_plot=='acf':
            #        ValueError('image_wrap plot type is not compatible with'
            #                      ' ACF, if this works axis values and limits '
            #                      'will be wrong')
                #x=x-max(x)/2
                #x=y-max(y)/2
            #    
            #    ax.imshow(Z, extent=[min(x),max(x),min(y),max(y)], aspect=1)
            else:
                msg=('Unsupported plot type:'+str(plot_type)+
                     ' please refer to documentation for supported types')
                ValueError(msg)
            
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
                                     self.grid_size)
                    y=np.abs(self.fft/self.profile.shape[0])
                    # line plot for 1d surfaces
                    ax.plot(x,y)
                    ax.xlim(0,max(x))
                else:
                    labels=['Scatter of frequency magnitudes', 
                            'frequency', '|F(x)|']
                    u=np.fft.fftfreq(self.profile.shape[0], 
                                     self.grid_size)
                    v=np.fft.fftfreq(self.profile.shape[1], 
                                     self.grid_size)
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
        else:
            msg=('Unsupported property to plot see documentation for details'
                 ', type given: ' + property_to_plot + 'supported types: ' 
                 + ' '.join(types2d+types1d))
            ValueError(msg)
            
    def __array__(self):
        """for easy compatability with numpy arrays"""
        return np.asarray(self.profile)

    def _get_points_from_extent(self, extent, spacing):
        if type(spacing) in [int, float, np.float32, np.float64]:
            spacing=[spacing]*2
        if type(extent[0]) in [int, float, np.float32, np.float64]:
            extent=[[0,extent[0]],[0,extent[1]]]
        x=np.arange(extent[0][0], extent[0][1], spacing[0])
        y=np.arange(extent[1][0], extent[1][1], spacing[1])
        
        X,Y=np.meshgrid(x,y)
        
        return(X,Y)
    
    def _interpolate(self,X,Y,**kwargs):
        if not kwargs is None:
            if 'remake' in kwargs:
                remake=kwargs['remake']
            else:
                remake=False
        if remake or not hasattr(self, 'interpolator'):
            try:
                import scipy.interpolate as interp
                self.interpolator=interp.interp2d(self.X,self.Y,self.profile, **kwargs)
            except AttributeError:
                msg="Surface must be descretised before interpolation"
                ValueError(msg)
        Z=self.interpolator(X,Y)
        return Z

if __name__=='__main__':
    A=Surface(file_name='C:\\Users\\44779\\code\\SlipPY\\image1_no header_units in nm', file_type='csv', delimitor=' ', grid_size=0.001)
    #testing show()
    #types2d=['profile', 'fft2d', 'psd', 'acf', 'apsd']
    #types1d=['histogram','fft1d', 'qq', 'disthist']
    #A.show(['profile','fft2d','acf'], ['surface', 'mesh', 'image'])
    #A.show(['histogram', 'fft1d', 'qq', 'disthist'])
    #out=A.birmingham(['sa', 'sq', 'ssk', 'sku'])
    #out=A.birmingham(['sds','sz', 'ssc'])
    #out=A.birmingham(['std','sbi', 'svi', 'str'])
    #out=A.birmingham(['sdr', 'sci', 'str'])
    