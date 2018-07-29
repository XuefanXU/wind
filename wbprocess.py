#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to program the MCP process.
Created on Thu May 31 15:04:48 2018

@author: XU Xuefan
"""
from scipy.special import gamma
import numpy as np
import pandas as pd

def ajtweibullsd(spd,sd,threshold_conv):
    k=1
    
    def a1(k):
        return spd/gamma(1+1/k)
    
    def a2(k):
        return ((sd**2+spd**2)/gamma(1+2/k))**0.5
    
    while True:
        if abs((a1(k)-a2(k))/min(a1(k),a2(k)))<=threshold_conv:
            break
        else:
            dydx=(a1(k+0.01)-a2(k+0.01)-a1(k)+a2(k))/0.01
            dk=abs((a1(k)-a2(k))/dydx)
            if a1(k)>a2(k):
                k=k+dk
            else:
                if k<dk:
                    k=0.5*k
                else:
                    k=k-dk
    return 0.5*(a1(k)+a2(k)),k

def ajtweibullcube(df_v,threshold_conv):
    mv=float(df_v.mean())
    mv3=float((df_v**3).mean())
    k=1
    
    def a1(k):
        return mv/gamma(1+1/k)
    
    def a3(k):
        return (mv3/gamma(1+3/k))**(1/3)

    while True:
        if abs((a1(k)-a3(k))/min(a1(k),a3(k)))<=threshold_conv:
            break
        else:
            dydx=(a1(k+0.01)-a3(k+0.01)-a1(k)+a3(k))/0.01
            dk=abs((a1(k)-a3(k))/dydx)
            if a1(k)>a3(k):
                k=k+dk
            else:
                if k<dk:
                    k=0.5*k
                else:
                    k=k-dk
    return 0.5*(a1(k)+a3(k)),k

def ajtweibullmlh(df_v,threshold_conv):
    k=1
    def a1(df_v,k):
        return (float(((df_v)**k).mean()))**(1/k)

    def a2(df_v,k):
        df_v1=np.log(df_v)
        df_v2=(df_v**k)*df_v1  
        return (float((df_v2.mean()))/(float(df_v1.mean())+1/k))**(1/k)
    
    while True:
        A1=a1(df_v,k)
        A2=a2(df_v,k)
        if abs(A1-A2)/min(A1,A2)<=threshold_conv:
            break
        else:
            dydx=(a1(df_v,k+0.01)-a2(df_v,k+0.01)-A1+A2)/0.01
            dk=abs((A1-A2)/dydx)
            if A1>A2:
                k=k+dk
            else:
                if k<dk:
                    k=0.5*k
                else:
                    k=k-dk
    return 0.5*(A1+A2),k