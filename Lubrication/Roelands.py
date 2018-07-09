# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 19:12:21 2018

@author: mike and lisa
"""

class Solver():
    updatables=[]
    
    def update_all(self):
        for solver in updatables:
            solver.update()
        self.update()

class RoelandsEquation(Solver):#some sort of general solver class? 
    """The dimentionless roelands equation. Takes a np.array as input 
    """
    def __init__(self, eta_o, p_h, p_o, roeland_co, roeland_z, pressure_giving_solver=False):
        self.eta_o=eta_o
        self.log_eta_o=np.log(eta_o)
        self.p_h=p_h
        self.p_o=p_o
        self.ph_over_po=p_h/p_o
        self.roeland_co=roeland_co
        self.roeland_z=roeland_z
        if pressure_giving_solver:
            self.p=pressure_giving_solver
            pressure_giving_solver.updateables.append(self)
        else:
            self.p=False
            
    def update(self, pressure=False):
        if self.p:
            pressures=self.p.pressure
            self.p.eta_no_dim=np.exp((self.log_eta_o+self.roeland_co)*
                             ((1+self.ph_over_po*pressures)**self.roeland_z-1))
            if type(pressure) is np.array:
                warnings.warn('Pressure specified and pressure giving solver'
                              'present, viscositys for the pressure giving'
                              ' solver are updated by default')
        elif type(pressure) is np.array:
            return np.exp((self.log_eta_o+self.roeland_co)*((1+self.ph_over_po
                          *pressure)**self.roeland_z-1))
            
class PressureDensity(Solver):
    
    def __init__(self, p_h, pd_coef_a=5.9e8, pd_coef_b=1.34, pressure_giving_solver=False):
        self.p_h=p_h
        self.pd_coef_a=pd_coef_a
        self.pd_coef_b=pd_coef_b
        if pressure_giving_solver:
            self.p=pressure_giving_solver
            pressure_giving_solver.updateables.append(self)
        else:
            self.p=False
            
    def update(self,pressure=False):
        if self.p:
            pressures=self.p.pressure
            self.p.rho_no_dim=(self.pd_coef_a+self.pd_coef_b*self.p_h*
                               pressures)/(self.pd_coef_a+self.p_h*pressures)
            if type(pressure) is np.array:
                warnings.warn('Pressure specified and pressure giving solver'
                              'present, densities for the pressure giving'
                              ' solver are updated by default')
        elif type(pressure) is np.array:
            return (self.pd_coef_a+self.pd_coef_b*self.p_h*pressure)/(
                    self.pd_coef_a+self.p_h*pressure)
            
class DiscretisedDeformationIntegral(Solver):#seems to only take pressure as input? ask Abdullah
    
    def __init__(self, )
   # solvers need upadatables, and update all function   
        
        ThisIsAClass
        this_is_a_var
        this_is_a_function()
        
#        
#pressure_solver=SomeSolverThatFindsPressureFromDensity()
##pressure_solver.pressures is the results
#density_solver=SomeSolverThatFindsDensityFromPressure(initial_density)
##as above
#
#pressure_solver.d=density_solver
#density_solver.p=pressure_solver
#
#while density_solver.error is big:
#    pressure_solver.update()#references self.d.density writes to self.pressure
#    density_solver.update()#references self.p.pressure writes to self.density
    