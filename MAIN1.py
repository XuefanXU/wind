#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script involves the main program to run the whole process of the project

Created on Fri May 25 11:34:53 2018

@author: XU Xuefan
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import sys
import random
import math as mt
from scipy.stats import pearsonr,spearmanr
import statsmodels.api as sm
import statsmodels.stats as ss
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.stats
import gc

import descriptive
import dataprocess
import wbprocess
import mcpprocess

csvpath_sit='../données/wind_LMS_treated20151001-20170930'
csvpath_sat='../données/Osterild-MERRA2'
headlines_sit=9
headlines_sat=24
columnindex_sat=[0,1,2,3]
columnnames_sat=['timestamp','speed_sat','direction_sat','temperature_sat'] #always put timestamp in the first place
timeformat='%d/%m/%Y %H:%M'
valman=-999

height_wanted=100
thres_nan=0.05
thres_p=0.05
thres_conv=0.02
thres_vcalm=0.5
thres_satcorr=0.75
thres_ev=0.75

def loaddata(csvpath):
    with open('%s.csv'%(csvpath)) as f:
        df=pd.read_csv(f,low_memory=False)
    return df

# def ajtweibullcube(df_v,threshold_conv):
#     mv=float(df_v.mean())
#     mv3=float((df_v**3).mean())
#     k=1
    
#     def a1(k):
#         return mv/gamma(1+1/k)
    
#     def a3(k):
#         return (mv3/gamma(1+3/k))**(1/3)

#     while True:
#         if abs((a1(k)-a3(k))/min(a1(k),a3(k)))<=threshold_conv:
#             break
#         else:
#             dydx=(a1(k+0.01)-a3(k+0.01)-a1(k)+a3(k))/0.01
#             dk=abs((a1(k)-a3(k))/dydx)
#             if a1(k)>a3(k):
#                 k=k+dk
#             else:
#                 if k<dk:
#                     k=0.5*k
#                 else:
#                     k=k-dk
#     return 0.5*(a1(k)+a3(k)),k
def presite():
    #############################
    #extract a pure data table from csv data
    #############################
    print('Extract pure data table from csv and save them into new csv file, besides generate a config json file.')
    dataprocess.seperate('%s.csv'%(csvpath_sit),headlines_sit,True)
    
    #############################
    #load csv to pandas dataframe
    #############################
    print('Load data from csv files')
    df_sit=loaddata(csvpath_sit+'_table')
    df_sit=df_sit.apply(pd.to_numeric,errors='ignore')
    
    #############################
    #descriptive of raw data_site
    #############################
    print('Statistique analysis of missing values in raw data set, saved in a csv file.')
    descriptive.stat_valman(df_sit,valman)
    #############################
    #descriptive of data withoout missing values
    #############################
    print('Descriptive analysis of site data without NaN, saved in another csv file.')
    df_sit[df_sit==valman]=np.nan
    df_sit.describe().to_csv('../outputs1/site_describe_nnan.csv')
    return df_sit

def ev_sit(df_sit):
    #############################
    #Select the site data whose height = the height of satellite data 
    #############################
    print('Select the site data (with direction mesurement) whose height = height_wanted, if not exist, do the vertical extrapolation.')
    configs_sit=dataprocess.config_read('../données/'+csvpath_sit.split('/')[-1]+'.json')
    dindex=np.where(abs(np.array(configs_sit['HAUTEURD']).astype(int)-height_wanted)==min(abs(np.array(configs_sit['HAUTEURD']).astype(int)-height_wanted)))[0][0]
    if str(height_wanted) in configs_sit['HAUTEURV']:
        vindex=configs_sit['HAUTEURV'].index(str(height_wanted))+1
        df_sit_ev=deepcopy(df_sit.loc[:,['Annees','Mois','Jours','Heures','Minutes','Spd'+str(vindex),'Dir'+str(dindex),'SD'+str(vindex)]])
        del dindex, vindex
    else:
        spd_cols=list(pd.Series(['Spd']*len(configs_sit['HAUTEURV']))+pd.Series(np.arange(len(configs_sit['HAUTEURV']))+1).astype(str))
        sd_cols=list(pd.Series(['SD']*len(configs_sit['HAUTEURV']))+pd.Series(np.arange(len(configs_sit['HAUTEURV']))+1).astype(str))    
        i=0
        while True:
            if configs_sit['HAUTEURV'][i]==configs_sit['HAUTEURV'][i+1]:
                del configs_sit['HAUTEURV'][i+1], spd_cols[i+1], sd_cols[i+1]
            i=i+1
            if i>=len(configs_sit['HAUTEURV'])-1:
                break
        del i
    #############################
    #Extrapolate wind speed to the needed height. 
    #############################
        print('Extrapolate wind speed to the needed height.')

        df_ev1=deepcopy(df_sit)
        df_ev1=df_ev1[df_ev1['Dir'+str(dindex+1)].isnull()==False]
        df_ev1=df_ev1[df_ev1[spd_cols].isnull().astype(int).sum(axis=1)<=len(spd_cols)-5]
            
        df_ev1=df_ev1.reset_index(drop=True)
        dict_model={'alphaEEN':[],'alpha':[],'alpha0.025':[],'alpha0.975':[],'intercept':[],'intercept0.025':[],
                    'intercept0.975':[],'rsquared':[],'pvalue_alpha':[],'pvalue_intercept':[],'aic':[],
                    'bic':[],'log_likelihood':[],'speed_reg':[],'speed_EEN':[]}
        for row in range(df_ev1.shape[0]):
            Y=df_ev1.loc[row,spd_cols][df_ev1.loc[row,spd_cols].isnull()==False]
            xindex=[]
            for i in list(Y.index):
                xindex.append(spd_cols.index(i))
            X=pd.Series(configs_sit['HAUTEURV']).iloc[xindex].astype(float)
            Y=Y.reset_index(drop=True)
            X=X.reset_index(drop=True)
            diff=abs(X-height_wanted)
            min1=diff.idxmin()
            diff[min1]=999
            min2=diff.idxmin()
            Y=np.log(Y)
            X=np.log(X)
            alphaeen=(Y[min1]-Y[min2])/(X[min1]-X[min2])
            dict_model['alphaEEN'].append(alphaeen)
            dict_model['speed_EEN'].append(mt.exp(alphaeen*(mt.log(height_wanted)-X[min1])+Y[min1]))
            row_model=lrm(X,Y,True)
            if row%(int(0.1*df_ev1.shape[0]))==0:
                row_model.save('../outputs1/ev/evmodel%d.pickle'%(int(row/int(0.1*df_ev1.shape[0]))))
            dict_model['alpha'].append(row_model.params.iat[1])
            dict_model['alpha0.025'].append(row_model.conf_int().iat[1,0])
            dict_model['alpha0.975'].append(row_model.conf_int().iat[1,1])
            dict_model['intercept'].append(row_model.params.iat[0])
            dict_model['intercept0.025'].append(row_model.conf_int().iat[0,0])
            dict_model['intercept0.975'].append(row_model.conf_int().iat[0,1])
            dict_model['rsquared'].append(row_model.rsquared)
            dict_model['pvalue_alpha'].append(row_model.pvalues.iat[1])
            dict_model['pvalue_intercept'].append(row_model.pvalues.iat[0])
            dict_model['aic'].append(row_model.aic)
            dict_model['bic'].append(row_model.bic)
            dict_model['log_likelihood'].append(row_model.llf)
            dict_model['speed_reg'].append(mt.exp(row_model.predict([1,mt.log(height_wanted)])[0]))                
            del i,xindex,X,Y,diff,alphaeen,min1,min2,row_model
            gc.collect()
        df_model=pd.DataFrame(dict_model,columns=['alphaEEN','alpha','alpha0.025','alpha0.975','intercept',
                                                  'intercept0.025','intercept0.975','rsquared','pvalue_alpha',
                                                  'pvalue_intercept','aic','bic','log_likelihood','speed_reg','speed_EEN'])
        df_model=df_ev1.loc[:,'Annees':'Minutes'].join(df_model,how='outer')
        df_model.loc[:,'Annees':'speed_EEN'].to_csv('../outputs1/ev/'+csvpath_sit.split('/')[-1]+'_ev1model.csv',index=None)
        
        
        
        
        df_model['speed_ev']=pd.concat([df_model[df_model.rsquared<thres_ev].speed_EEN,df_model[df_model.rsquared>=thres_ev].speed_reg])
        df_sit_ev=df_model[['Annees','Mois','Jours','Heures','Minutes','speed_ev']]
        df_sit_ev['direction']=deepcopy(df_ev1['Dir'+str(dindex+1)])
        df_sit_ev['std']=((df_ev1[sd_cols]**2).mean(axis=1))**0.5
        del row,dict_model,dindex
        gc.collect()

    df_sit_ev.columns=['Annees','Mois','Jours','Heures','Minutes','speed_sit','direction_sit','std_sit']

    #############################
    #drop rows containing any NaN value when the ratio (NaN number/total number) is inferior to one threshold determined by expert 
    #############################
    print('Check NaN values in dataset')
    nan_percent=(df_sit.shape[0]-df_sit_ev.shape[0])/df_sit.shape[0]
    if nan_percent<=thres_nan:
        print('NaN values occupy %f%% <= %f%% of the total, therfore drop them.'%(nan_percent*100,thres_nan*100))
    else:
        print('NaN values occupy %f%% > %f%% of the total, process has to stop until an algorithm to fill those NaN has been developped.'%(nan_percent*100,thres_nan*100))
        sys.exit()    
    df_sit_ev.to_csv('../outputs1/ev/%s_%dm.csv'%(csvpath_sit.split('/')[-1],height_wanted),index=None)

    #############################
    #pick some models and visualize them. You can comment the codes below if you do not want to do that.
    #############################
    for i in range(11):
        model=sm.load('../outputs1/ev/evmodel%d.pickle'%(i))
        row=i*int(0.1*df_ev1.shape[0])
        Y_real=df_ev1.loc[row,spd_cols][df_ev1.loc[row,spd_cols].isnull()==False]
        xindex=[]
        for j in list(Y_real.index):
            xindex.append(spd_cols.index(j))
        X=pd.Series(configs_sit['HAUTEURV']).iloc[xindex].astype(float)
        X=X.reset_index(drop=True)
        logX=np.log(X)
        Y_real=Y_real.reset_index(drop=True)
        logY_real=np.log(Y_real)
        diff=abs(X-height_wanted)
        min1=diff.idxmin()
        Y_EEN=df_model.at[row,'speed_EEN']
        logY_EEN=mt.log(Y_EEN)
        pre_std,logY_l,logY_u=wls_prediction_std(model)
        Y_l=np.exp(logY_l)
        Y_u=np.exp(logY_u)
        residmean=model.resid.mean()
        residstd=model.resid.std(ddof=0)
        resid_norm=(model.resid-residmean)/residstd
        time=pd.to_datetime(df_ev1.at[row,'Jours'].astype(str)+'/'+df_ev1.at[row,'Mois'].astype(str)+'/'+df_ev1.at[row,'Annees'].astype(str)+' '+df_ev1.at[row,'Heures'].astype(str)+':'+df_ev1.at[row,'Minutes'].astype(str),format='%d/%m/%Y %H:%M')
        print('alphaEEN = ',df_model.at[row,'alphaEEN'])
        print('alphareg = ',df_model.at[row,'alpha'])
        print('IC alpha = ',[df_model.at[row,'alpha0.025'],df_model.at[row,'alpha0.975']])
        print('R2 = ',df_model.at[row,'rsquared'])
        descriptive.ev_fit(X,Y_real,np.exp(model.fittedvalues),Y_EEN,Y_l,Y_u,height_wanted,time,i,False,1)
        descriptive.ev_fit(logX,logY_real,model.fittedvalues,logY_EEN,logY_l,logY_u,mt.log(height_wanted),time,i,True,1)
        descriptive.ev_residu(logX,model.resid,time,i,1)
        descriptive.ev_qqplot(resid_norm,time,i,1)
        del model,row,Y_real,xindex,j,X,logX,logY_real,diff,min1,logY_EEN,Y_EEN,pre_std,logY_l,logY_u,Y_l,Y_u,residmean,residstd,resid_norm,time       
        gc.collect()
    descriptive.ev_boxplot(df_model,1)
    descriptive.ev_curve(df_model,1)
    descriptive.ev_histo(df_model,1)

def mcp_sit(df_sit_mcp):   
    #############################
    #calculate the hourly expectations for site data
    #############################
    print('Calculate the hourly expectations for site data,using weibull distribution law')
    dict_sit_mcp_hour=[]
    v_minute=[]
    theta_minute=[]
    for i in range(df_sit_mcp.shape[0]):
        spd=df_sit_mcp.at[i,'speed_sit']
        sd=df_sit_mcp.at[i,'std_sit']
        ia,ik=wbprocess.ajtweibullsd(spd,sd,thres_conv)
        for j in range(200):
            v_minute.append(random.weibullvariate(ia,ik))
        theta_rand=mt.radians(df_sit_mcp.at[i,'direction_sit']-180)
        theta_minute.append([mt.cos(theta_rand),mt.sin(theta_rand)])
        if i==df_sit_mcp.shape[0]-1:
            dict_sit_mcp_hour.append({'Annees':df_sit_mcp.at[i,'Annees'],
                                    'Mois':df_sit_mcp.at[i,'Mois'],
                                    'Jours':df_sit_mcp.at[i,'Jours'],
                                    'Heures':df_sit_mcp.at[i,'Heures'],
                                    'speed_sit':mcpprocess.meanspeed(v_minute)[0],
                                    'direction_sit':mcpprocess.meandirec(theta_minute),
                                    'stdv_sit':mcpprocess.meanspeed(v_minute)[1],
                                    'speed3_sit':mcpprocess.meanspeed(v_minute)[2],
                                    'stdv3_sit':mcpprocess.meanspeed(v_minute)[3]})
            del v_minute, theta_minute,i, ia, ik, j, sd, spd,theta_rand
        else:
            if np.any(df_sit_mcp.loc[i,'Annees':'Heures']!=df_sit_mcp.loc[i+1,'Annees':'Heures']):
                dict_sit_mcp_hour.append({'Annees':df_sit_mcp.at[i,'Annees'],
                                        'Mois':df_sit_mcp.at[i,'Mois'],
                                        'Jours':df_sit_mcp.at[i,'Jours'],
                                        'Heures':df_sit_mcp.at[i,'Heures'],
                                        'speed_sit':mcpprocess.meanspeed(v_minute)[0],
                                        'direction_sit':mcpprocess.meandirec(theta_minute),
                                        'stdv_sit':mcpprocess.meanspeed(v_minute)[1],
                                        'speed3_sit':mcpprocess.meanspeed(v_minute)[2],
                                        'stdv3_sit':mcpprocess.meanspeed(v_minute)[3]})
                del v_minute[:],theta_minute[:]
    mcp_hour_cols=list(df_sit_mcp.columns.drop(['Minutes','std_sit']))
    mcp_hour_cols.extend(['stdv_sit','speed3_sit','stdv3_sit'])
    df_sit_mcp_hour=pd.DataFrame(dict_sit_mcp_hour,columns=mcp_hour_cols)
    df_sit_mcp_hour.to_csv('../outputs1/ev/%s_mcphour.csv'%(csvpath_sit.split('/')[-1]),index=None)
    print('The hourly site data is saved in a csv file.')

def presat():
    #############################
    #extract a pure data table from csv data
    #############################
    print('Extract pure data tables from csv and save them into new csv files.')
    dataprocess.seperate('%s.csv'%(csvpath_sat),headlines_sat,False)
    
    #############################
    #load csv to pandas dataframe
    #############################
    print('Load data from csv files.')
    df_sat=loaddata(csvpath_sat+'_table')
    
    #############################
    #extract requisite columns from the satellite dataframe
    #############################
    df_sat=df_sat.iloc[1:,columnindex_sat]
    df_sat=df_sat.reset_index(drop=True)
    df_sat.columns=columnnames_sat
    
    #############################
    #divide the timestamp into 5 columns of year, month, day, hour and minute
    #############################
    print('Divide the timestamp of satellite data into 5 columns.')
    df_sat['timestamp']=pd.to_datetime(df_sat.timestamp,format=timeformat)
    dataprocess.timecut_sat(df_sat)
    df_sat[columnnames_sat[1:]] = df_sat[columnnames_sat[1:]].apply(pd.to_numeric,errors='ignore')
    df_sat['speed3_sat']=df_sat.speed_sat**3
    return df_sat

def lrm(X,Y,addconstant=True):
    if addconstant:
        X=sm.add_constant(X)
    model=sm.OLS(Y,X).fit()
    return model

def mcp_vis():
    print('Pre-process for satellite data:')
    df_sat=presat()
    ##############################
    #concatenation between site and satellite
    ##############################
    print('Concatenate the satellite data and the site data.')
    df_sat_mcp_hour=df_sat.drop('Minutes',axis=1)# run this line only if the hourly-expextation-calculation process is not applied to the satellite data
    df_sit_mcp_hour=loaddata('../outputs1/%s_mcphour'%(csvpath_sit.split('/')[-1]))
    mcpprocess.timemerge(df_sat_mcp_hour,df_sit_mcp_hour,['Annees','Mois','Jours','Heures'],1,csvpath_sat.split('/')[-1]+'-'+csvpath_sit.split('/')[-1])
    print('Data sat-sit is saved in a csv file.')
    
    ##############################
    #visualisation of the description of the sat_sit data
    ##############################
    print('Load sat-sit data.')
    df_sat_sit=loaddata('../outputs1/rl/'+csvpath_sat.split('/')[-1]+'-'+csvpath_sit.split('/')[-1])
    print('Descriptive analysis - long term speed boxplot')
    descriptive.dfboxplot(df_sat_sit[['speed_sat','speed_sit']],1)
    print('Descriptive analysis - long term speed evolution')
    descriptive.dfcurve(df_sat_sit[['timestamp','speed_sat','speed_sit']],'Long-term',1)
    print('Descriptive analysis - long term speed cube boxplot')
    descriptive.dfboxplot(df_sat_sit[['speed3_sat','speed3_sit']],1)
    print('Descriptive analysis - long term speed cube evolution')
    descriptive.dfcurve(df_sat_sit[['timestamp','speed3_sat','speed3_sit']],'Long-term',1)
    ##############################
    #Extract the recoverred rows and filter the calm winds
    ##############################
    print('Extract recoverred rows, filter calm winds and seperate the direction sectors')
    df_short=df_sat_sit.dropna(axis=0, how='any')
    df_short=mcpprocess.calmwind(df_short,thres_vcalm)
    df_sector=mcpprocess.dirsector(df_short[['direction_sat','direction_sit']])
    df_sector.columns=['sector_sat','sector_sit']
    df_short=pd.merge(df_short,df_sector,how='outer',left_index=True,right_index=True)
    del df_sector
    gc.collect()
    print('Descriptive analysis - short term speed boxplot')
    descriptive.dfboxplot(df_short[['speed_sat','speed_sit']],1)
    print('Descriptive analysis - short term speed evolution')
    descriptive.dfcurve(df_short[['timestamp','speed_sat','speed_sit']],'Short-term',1)
    print('Descriptive analysis - short term speed cube boxplot')
    descriptive.dfboxplot(df_short[['speed3_sat','speed3_sit']],1)
    print('Descriptive analysis - short term speed cube evolution')
    descriptive.dfcurve(df_short[['timestamp','speed3_sat','speed3_sit']],'Short-term',1)    
    descriptive.sec_histo(df_short[['sector_sat','sector_sit']],1)
    df_short['windveer']=df_short['direction_sit']-df_short['direction_sat']+360*(df_short['direction_sit']-df_short['direction_sat']<=-180).astype(int)-360*(df_short['direction_sit']-df_short['direction_sat']>180).astype(int)
    for i in range(1,13):
        df_short[df_short.sector_sat==i].to_csv('../outputs1/rl/regression_sector%d.csv'%(i),index=None)
    del i

    #############################
    #Linear regression modelling by sector
    #############################
    print('Plotting by sector:')
    for i in range(1,13):
        df_regression=loaddata('../outputs1/rl/regression_sector%d'%(i))
        descriptive.dfboxplot(df_regression[['speed_sat','speed_sit']],1,df_regression.at[0,'sector_sat'])
        descriptive.dfboxplot(pd.DataFrame(df_regression['windveer']),1,df_regression.at[0,'sector_sat'])
        descriptive.dfboxplot(df_regression[['speed3_sat','speed3_sit']],1,df_regression.at[0,'sector_sat'])

        descriptive.dfcurve(df_regression[['timestamp','speed_sat','speed_sit']],'ST',1,df_regression.at[0,'sector_sat'])
        descriptive.dfcurve(df_regression[['timestamp','windveer']],'ST',1,df_regression.at[0,'sector_sat'])
        descriptive.dfcurve(df_regression[['timestamp','speed3_sat','speed3_sit']],'ST',1,df_regression.at[0,'sector_sat'])
        descriptive.sec_histo(pd.DataFrame(df_regression['sector_sit']),1,df_regression.at[0,'sector_sat'])
    
        descriptive.dfscatter(df_regression,1,'speed_sat','speed_sit')
        descriptive.dfscatter(df_regression,1,'speed_sat','windveer')
        descriptive.dfscatter(df_regression,1,'speed_sit','windveer')
    del i, df_regression
    
    
def mcp_model(df_sector,sector,thres_v):
#    print('Check the correlation between site and satellite:')
#    corr_sit_sat=pearsonr(df_regression['speed_sit'],df_regression['speed_sat'])
#    if corr_sit_sat[0]<threshold_satcorr or corr_sit_sat[1]>threshold_p:
#        print("Pearson's test p-value is %f %s %f, correlation between valuables is %f %s %f, therefore this satellite data should be rejected."%(corr_sit_sat[1],'>' if corr_sit_sat[1]>threshold_p else '<=',threshold_p,corr_sit_sat[0],'<' if corr_sit_sat[0]<threshold_satcorr else '>=',threshold_satcorr))
#        sys.exit()
#    else:
#        print("Pearson's test p-value is %f <= %f, and the correlation between valuables is %f >= %f"%(corr_sit_sat[1],threshold_p,corr_sit_sat[0],threshold_satcorr))
#    corr3_sit_sat=pearsonr(df_regression['speed3_sit'],df_regression['speed3_sat'])
#    if corr3_sit_sat[0]<threshold_satcorr or corr3_sit_sat[1]>threshold_p:
#        print("Pearson's test p-value is %f %s %f, correlation between valuables is %f %s %f, therefore this satellite data should be rejected."%(corr3_sit_sat[1],'>' if corr3_sit_sat[1]>threshold_p else '<=',threshold_p,corr3_sit_sat[0],'<' if corr3_sit_sat[0]<threshold_satcorr else '>=',threshold_satcorr))
#        sys.exit()
#    else:
#        print("Pearson's test p-value is %f <= %f, and the correlation between valuables is %f >= %f"%(corr3_sit_sat[1],threshold_p,corr3_sit_sat[0],threshold_satcorr))
#    
#    print('Linear regression (speed_sit, veer) ~ speed_sat:')
#    print('Cross validation process.')
#    df_train,df_test=mcpprocess.holdout(df_regression,0.75)
#    mcpmodel=lrm(df_train['speed_sat'],df_train['speed_sit'])
#    print(mcpmodel.summary2())
    df_regression=df_sector[df_sector.speed_sit>thres_v].reset_index(drop=True)
    #train:test=4:1
    df_train,df_test=mcpprocess.holdout(df_regression,0.2)
           
    pre_std,Y_l,Y_u=wls_prediction_std(mcpmodel_v)

    descriptive.rl_fit(df_regression.speed_sat,df_regression.speed_sit,mcpmodel_v.fittedvalues,Y_l,Y_u,sector,True,False,1)
    descriptive.rl_residu(df_regression.speed_sat,mcpmodel_v.resid,sector,True,False,1)
    descriptive.rl_qqplot(resid_norm,sector,True,False,1)


def main():
    '''
    Process order:
        load site data;
        vertical extrapolation to height_sat;
        long terme reconstruction:
            1. mcp methode:
                load satellite data;
                calculate hourly expectation of site data;
                linear regression model by sector;
        vertical extrapolation to the needed height                
    '''
    df_sit=presite()
    ev_sit(df_sit)
    
    df_sit_mcp=loaddata('../outputs1/ev/%s_%dm'%(csvpath_sit.split('/')[-1],height_wanted))
    mcp_sit(df_sit_mcp)
    mcp_vis()
    
    sector=1
    df_sector=loaddata('../outputs1/rl/regression_sector%d'%(sector))
    thres_v=0            
    mcp_model(df_sector,sector,thres_v)
    

    #############################
    #alalyse the correlation between speed and direction
    #############################
    # print('Check the dependance between speed and direction')
    # sit_corr=descriptive.corr_test(df_sit_mcp['speed_sit'],df_sit_mcp['direction_sit'])
    # sat_corr=descriptive.corr_test(df_sat['speed_sat'],df_sat['direction_sat'])
    # if sit_corr[1]<threshold_p:
    #     print("p-value of Pearson's test = %f < %f, therfore valuables speed and direction are independant."%(sit_corr[1],threshold_p))
    # else:
    #     print("p-value of Pearson's test = %f >= %f, therfore valuables speed and direction are dependant, their correlation is %f"%(sit_corr[1],threshold_p,sit_corr[0]))
    #descriptive.dfcurve(df_sat_sit.loc[:,['timestamp','windveer']])    
if __name__=='__main__':
    #main()
    pass