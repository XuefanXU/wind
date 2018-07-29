#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to program the MCP process
Created on Mon Jun  4 14:26:40 2018

@author: XU Xuefan
"""

import pandas as pd
import numpy as np
import math as mt
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt

def timemerge(df_sat,df_sit,onkeys,senario,dfname):
    df_sat_sit=pd.merge(df_sat, df_sit, how='outer', on=onkeys)
    df_sat_sit.insert(0,'timestamp',df_sat_sit['Jours'].astype(str)+'/'+df_sat_sit['Mois'].astype(str)+'/'+df_sat_sit['Annees'].astype(str)+' '+df_sat_sit['Heures'].astype(str))
    df_sat_sit['timestamp']=pd.to_datetime(df_sat_sit.timestamp,format='%d/%m/%Y %H')
    df_sat_sit.to_csv('../outputs%d/rl/%s.csv'%(senario,dfname),index=None)

def meanspeed(v_minute):
    v_minute=np.array(v_minute,dtype=np.float32)
    v3_minute=v_minute**3
    return float(v_minute.mean()), float(v_minute.std(ddof=0)),float(v3_minute.mean()),float(v3_minute.std(ddof=0))

def meandirec(theta_minute):
    theta_minute=np.array(theta_minute,dtype=np.float32)
    x,y=theta_minute.mean(axis=0)
    if x!=0 or y!=0:
        return mt.degrees(mt.atan2(y,x)+mt.pi)
    else:
        x,y=random.choice(theta_minute)
        return mt.degrees(mt.atan2(y,x)+mt.pi)

def dirsector(df_direc):
    """
    all directions are divided into 12 sectors; the range of sectors are defined as: [345,15),[15,45),[45,75)...
    """
    df_sector=((df_direc+15-360*(df_direc>=345).astype(int))//30+1).astype(int)
    #df_sector.columns=['sector_%s'%('sat' if list(df_direc.columns)==['direction_sat'] else 'sit')]
    return df_sector

def calmwind(df_regression,threshold_vcalm):
    df_regression=df_regression[(df_regression[['speed_sat','speed_sit']]>=threshold_vcalm).any(axis=1)]
    df_regression=df_regression.reset_index(drop=True)
    df_notcalm=pd.DataFrame(df_regression.direction_sat[df_regression.speed_sat>=threshold_vcalm])
    df_sector=dirsector(df_notcalm)
    del df_notcalm
    df_sector.columns=['sector_sat']
    distri_sat=np.histogram(df_sector.sector_sat,bins=list(np.arange(1,14)),density=True)[0]
    distri_sat=distri_sat.cumsum()*10000    
    vcalm=[]
    for i in range(df_regression.shape[0]):
        if df_regression.at[i,'speed_sat']<threshold_vcalm:
            vcalm.append(i)
        else:
            if vcalm!=[]:
                if len(vcalm)<10:
                    if vcalm[0]==0:
                        jsector=df_sector.at[vcalm[-1]+1,'sector_sat']
                    elif vcalm[-1]==df_regression.shape[0]-1:
                        jsector=df_sector.at[vcalm[0]-1,'sector_sat']
                    else:
                        jsector=(df_sector.at[vcalm[0]-1,'sector_sat']+df_sector.at[vcalm[-1]+1,'sector_sat'])/2
                    for j in vcalm:
                        jdirection=random.randint(300*jsector-450+3600*int(jsector==1),300*jsector-151+3600*int(jsector==1))
                        jdirection=(jdirection-3600*int(jdirection>3600))*0.1
                        df_regression.at[j,'direction_sat']=jdirection                    
                else:
                    for j in vcalm:
                        randsector=random.randint(0,10000)
                        pivot1=0
                        pivot2=len(distri_sat)
                        jsector=int(pivot2/2)
                        while jsector!=0:
                            if randsector<=distri_sat[jsector] and randsector>distri_sat[jsector-1]:
                                break
                            else:
                                if randsector<=distri_sat[jsector-1]:
                                    pivot2=jsector
                                else:
                                    pivot1=jsector
                                jsector=int((pivot1+pivot2)/2)
                        jdirection=random.randint(300*jsector-150+3600*int(jsector==0),300*jsector+149+3600*int(jsector==0))
                        jdirection=(jdirection-3600*int(jdirection>3600))*0.1
                        df_regression.at[j,'direction_sat']=jdirection
                del vcalm[:]
    del df_sector
    return df_regression

def holdout(df_regression,fraction):
    df_train=df_regression.iloc[int(df_regression.shape[0]*fraction):,:].reset_index(drop=True)
    df_test=df_regression.iloc[:int(df_regression.shape[0]*fraction),:]
    return df_train,df_test

def pltregression():
    pass    
