# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:08:14 2019

@author: Michael
"""

class A(object):
    
    _my_list=[]
    def __init__(self, value):
        self._my_list=[value]
    
    @property
    def my_list_property(self):
        return(self._my_list)
    
    @my_list_property.setter
    def my_list_property(self, value):
        self._my_list=value
        
    @my_list_property.deleter
    def my_list_property(self):
        self._my_list=[]

if __name__=='__main__':
    a=A(1)
    print(a.my_list_property)
    [1]
    a.my_list_property.append(2)
    print(a.my_list_property)
    [1,2]