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