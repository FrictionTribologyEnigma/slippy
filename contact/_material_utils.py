import numpy as np

__all__=['_Bunch', '_get_properties']

class _Bunch(object):
  """A class to tidy dict indexing: (my_dict['e']) into attribute access:
  my_class.e, makes equations shorter and more understandable    
  """
    
  def __init__(self, adict):
    self.__dict__.update(adict)
    
def _get_properties(set_props: dict):
    """Get all elastic properties from any pair
    
    Parameters
    ----------
    set_props : dict
        dict of properties must have exactly 2 members valid keys are: 'K', 
        'E', 'v', 'Lam', 'M', 'G'
    
    Returns
    -------
    out : dict
        dict of all material properties keys are: 'K', 'E', 'v', 'Lam', 'M', 'G'
    
    Notes
    -----
    
    Keys refer to:
        - E - Young's modulus
        - v - Poission's ratio
        - K - Bulk Modulus
        - Lam - Lame's first parameter
        - G - Shear modulus
        - M - P wave modulus 
    
    """
    if len(set_props)!=2:
        raise ValueError("Exactly 2 properties must be set,"
                         " {} found".format(len(set_props))) 
    
    valid_keys=['K', 'E', 'v', 'G', 'Lam', 'M']
    
    set_params=[key for key in list(set_props.keys()) if key in valid_keys]
    
    if len(set_params)!=2:
        msg=("Invalid keys in set_props keys found are: " +
             "{}".format(set_props.keys()) +
             ". Valid keys are: "+" ".join(valid_keys))
        raise ValueError(msg)
    
    out=set_props.copy()
    
    set_params=list(set_props.keys())
    set_params.sort()
    #p is properties this saves a lot of space
    p=_Bunch(set_props)
    
    if set_params[0]=='E':
        if set_params[1]=='G':
            out['K']=p.E*p.G/(3*(3*p.G-p.E))
            out['Lam']=p.G*(p.E-2*p.G)/(3*p.G-p.E)
            out['M']=p.G*(4*p.G-p.E)/(3*p.G-p.E)
            out['v']=p.E/(2*p.G)-1
        elif set_params[1]=='K':
            out['G']=3*p.K*p.E/(9*p.K-p.E)
            out['Lam']=3*p.K*(3*p.K-p.E)/(9*p.K-p.E)
            out['M']=3*p.K*(3*p.K+p.E)/(9*p.K-p.E)
            out['v']=(3*p.K-p.E)/(6*p.K)
        elif set_params[1]=='Lam':
            R=np.sqrt(p.E**2+9*p.Lam**2+2*p.E*p.Lam)
            out['G']=(p.E-3*p.Lam+R)/4
            out['K']=(p.E+3*p.Lam+R)/6
            out['M']=(p.E-p.Lam+R)/2
            out['v']=2*p.Lam/(p.E+p.Lam+R)
        elif set_params[1]=='M':
            S=np.sqrt(p.E**2+9*p.M**2-10*p.E*p.M)
            out['G']=(3*p.M+p.E-S)/8
            out['K']=(3*p.M-p.E+S)/6
            out['Lam']=(p.M-p.E+S)/4
            out['v']=(p.E-p.M+S)/(4*p.M)
        else: # set_params[1]=='v'
            out['G']=p.E/(2*(1+p.v))
            out['K']=p.E/(3*(1-2*p.v))
            out['Lam']=p.E*p.v/((1+p.v)*(1-2*p.v))
            out['M']=p.E*(1-p.v)/((1+p.v)*(1-2*p.v))
    elif set_params[0]=='G':
        if set_params[1]=='K':
            out['E']=9*p.K*p.G/(3*p.K+p.G)
            out['Lam']=p.K-2*p.G/3
            out['M']=p.K+4*p.G/3
            out['v']=(3*p.K-2*p.G)/(2*(3*p.K+p.G))
        elif set_params[1]=='Lam':
            out['E']=p.G*(3*p.Lam+2*p.G)/(p.Lam+p.G)
            out['K']=p.Lam+2*p.G/3
            out['M']=p.Lam+2*p.G
            out['v']=p.Lam/(2*(p.Lam+p.G))
        elif set_params[1]=='M':
            out['E']=p.G*(3*p.M-4*p.G)/(p.M-p.G)
            out['K']=p.M-4*p.G/3
            out['Lam']=p.M-2*p.G
            out['v']=(p.M-2*p.G)/(2*p.M-2*p.G)
        else: # set_params[1]=='v'
            out['E']=2*p.G*(1+p.v)
            out['K']=2*p.G*(1+p.v)/(3*(1-2*p.v))
            out['Lam']=2*p.G*p.v/(1-2*p.v)
            out['M']=2*p.G*(1-p.v)/(1-2*p.v)
    elif set_params[0]=='K':
        if set_params[1]=='Lam':
            out['E']=9*p.K*(p.K-p.Lam)/(3*p.K-p.Lam)
            out['G']=3*(p.K-p.Lam)/2
            out['M']=3*p.K-2*p.Lam
            out['v']=p.Lam/(3*p.K-p.Lam)
        elif set_params[1]=='M':
            out['E']=9*p.K*(p.M-p.K)/(3*p.K+p.M)
            out['G']=3*(p.M-p.K)/4
            out['Lam']=(3*p.K-p.M)/2
            out['v']=(3*p.K-p.M)/(3*p.K+p.M)
        else: # set_params[1]=='v'
            out['E']=3*p.K*(1-2*p.v)
            out['G']=(3*p.K*(1-2*p.v))/(2*(1+p.v))
            out['Lam']=3*p.K*p.v/(1+p.v)
            out['M']=3*p.K*(1-p.v)/(1+p.v)
    elif set_params[0]=='Lam':
        if set_params[1]=='M':
            out['E']=(p.M-p.Lam)*(p.M+2*p.Lam)/(p.M+p.Lam)
            out['G']=(p.M-p.Lam)/2
            out['K']=(p.M+2*p.Lam)/3
            out['v']=p.Lam/(p.M+p.Lam)
        else:
            out['E']=p.Lam*(1+p.v)*(1-2*p.v)/p.v
            out['G']=p.Lam(1-2*p.v)/(2*p.v)
            out['K']=p.Lam*(1+p.v)/(3*p.v)
            out['M']=p.Lam*(1-p.v)/p.v
    else:
        out['E']=p.M*(1+p.v)*(1-2*p.v)/(1-p.v)
        out['G']=p.M*(1-2*p.v)/(2*(1-p.v))
        out['K']=p.M*(1+p.v)/(3*(1-p.v))
        out['Lam']=p.M*p.v/(1-p.v)
    
    return out