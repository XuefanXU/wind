#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to separate in the csv file the patrameters at the header and the data array, and to generate a new csv file 
to save the pure data array, a json file to save the parameters.
 
Created on Fri Apr 27 10:10:09 2018

@author: XU Xuefan
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import json

def seperate(csvpath,beforehead,site):
    databrut=np.genfromtxt(csvpath,delimiter=",",dtype=None).astype(str)
    datas=deepcopy(databrut[beforehead:])
    dataname=csvpath.split('/')[-1].split('.')[0]
    if site:
        configs=deepcopy(databrut[:beforehead])
        config_save(configs,dataname)
    del databrut
    df=pd.DataFrame(datas)
    df.to_csv('../données/%s_table.csv'%(dataname),index=None,header=None)
    
        
def config_save(configs,dataname):
    configdict={}
    for configrow in configs:
        if configrow[0]!='':
            configdict[configrow[0]]=[]
            for i in range(1,len(configrow)):
                if configrow[i]!='':
                    configrow[i].strip()
                    configdict[configrow[0]].append(configrow[i])    
    with open('../données/%s.json'%(dataname),'w') as f:
        json.dump(configdict,f)

def config_read(jsfile):
    with open(jsfile,'r') as f:
        data=json.load(f)
    return data

def timecut_sat(df_sat):
    df_sat.insert(1,'Annees',df_sat['timestamp'].dt.year)
    df_sat.insert(2,'Mois',df_sat['timestamp'].dt.month)
    df_sat.insert(3,'Jours',df_sat['timestamp'].dt.day)
    df_sat.insert(4,'Heures',df_sat['timestamp'].dt.hour)
    df_sat.insert(5,'Minutes',df_sat['timestamp'].dt.minute)
    df_sat.pop('timestamp')

def main():
    csvpath_sit='../données/wind_LMS_treated20151001-20170930.csv'
    csvpath_sat='../données/Osterild-MERRA2.csv'
    seperate(csvpath_sit,9,True)
    seperate(csvpath_sat,24,False)

if __name__=='__main__':
    main()    