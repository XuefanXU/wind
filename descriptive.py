#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is used to analyze the data description and visualize it.
Created on Fri May  4 11:25:51 2018

@author: XU Xuefan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def stat_valman(df_sit,valman):
    sit_lines=np.shape(df_sit)[0]
    df_nan=pd.DataFrame(np.zeros((len(df_sit.columns),2)),index=df_sit.columns,columns=['nbNAN','pourcentNAN(%)'])
    for i in df_sit.columns:
        df_nan.at[i,'nbNAN']=np.shape(df_sit[i][df_sit[i]==valman])[0]
        df_nan.at[i,'pourcentNAN(%)']=100*np.shape(df_sit[i][df_sit[i]==valman])[0]/sit_lines
    df_nan.to_csv('../outputs1/site_valmanq.csv')

def dfboxplot(df,senario,sector_sat=None):
    p=df.boxplot(column=list(df.columns),showmeans=True)
    if 'speed3' in list(df.columns)[0]:
        p.set_ylabel('v^3 (m/s)^3')
        if sector_sat is None:
            p.set_title('Descriptive analysis of speed cube for the satellite and the site')
            plt.savefig('../outputs%d/rl/boxplot_speed3_sat-sit.png'%(senario))
        else:
            p.set_title('Descriptive analysis of speed cube for the site, as sector_sat=%d'%(sector_sat))
            plt.savefig('../outputs%d/rl/boxplot_speed3_sit-sat=%d.png'%(senario,sector_sat))
    elif 'windveer' in list(df.columns):
        p.set_ylabel('Δθ (°)')
        p.set_title('Descriptive analysis of wind veer when sector_sat=%d'%(sector_sat))
        plt.savefig('../outputs%d/rl/boxplot_veer_sit_sat=%d.png'%(senario,sector_sat))
    else:
        p.set_ylabel('v (m/s)')
        if sector_sat is None:
            p.set_title('Descriptive analysis of wind speed for the satellite and the site')
            plt.savefig('../outputs%d/rl/boxplot_speed_sat-sit.png'%(senario))
        else:
            p.set_title('Descriptive analysis of wind speed for the site, as sector_sat=%d'%(sector_sat))
            plt.savefig('../outputs%d/rl/boxplot_speed_sit-sat=%d.png'%(senario,sector_sat))
    plt.show()

def dfcurve(df,period,senario,sector_sat=None):
    if 'speed3' in list(df.columns)[1]:
        if sector_sat is None:
            p=df.plot(title='%s evolution of speed cube for the satellite and the site'%(period),legend=True)
            p.set_ylabel('v^3 (m/s)^3')
            plt.legend(loc ='best')
            plt.savefig('../outputs%d/rl/%sevolution_speed3_sat-sit.png'%(senario,period))
        else:
            p=df.plot(title='%s evolution of speed cube for the site, as sector_sat=%d'%(period,sector_sat),legend=True)
            p.set_ylabel('v^3 (m/s)^3')
            plt.legend(loc ='best')
            plt.savefig('../outputs%d/rl/%sevolution_speed3_sit-sat=%d.png'%(senario,period,sector_sat))
    elif 'windveer' in list(df.columns):
        p=df.plot(title='%s evolution of wind veer for the site, as sector_sat=%d'%(period,sector_sat),legend=True)
        p.set_ylabel('Δθ (°)')
        plt.legend(loc ='best')
        plt.savefig('../outputs%d/rl/%sevolution_veer_sat=%d.png'%(senario,period,sector_sat))
    else:
        if sector_sat is None:
            p=df.plot(title='%s evolution of wind speed for the satellite and the site'%(period),legend=True)
            p.set_ylabel('v (m/s)')
            plt.legend(loc ='best')
            plt.savefig('../outputs%d/rl/%sevolution_speed_sat-sit.png'%(senario,period))
        else:
            p=df.plot(title='%s evolution of wind speed for the site, as sector_sat=%d'%(period,sector_sat),legend=True)
            p.set_ylabel('v (m/s)')
            plt.legend(loc ='best')
            plt.savefig('../outputs%d/rl/%sevolution_speed_sit-sat=%d.png'%(senario,period,sector_sat)) 
    plt.show()

def sec_histo(df,senario,sector_sat=None):    
    if 'sector_sat' in list(df.columns):
        df.hist(bins=list(np.arange(1,14)),density=True,sharey=True)
        plt.savefig('../outputs%d/rl/sector_sat-sit.png'%(senario))
    else:
        df.hist(bins=list(np.arange(1,14)),density=True)
        plt.savefig('../outputs%d/rl/sector_sit-sat=%d.png'%(senario,sector_sat))
    plt.show()
    
def dfscatter(df,senario,xlab,ylab):
    p=df.plot(x=xlab,y=ylab,kind='scatter',legend=True,title=ylab+' vs '+xlab+', as sector_sat='+str(df.sector_sat.iat[0]))
    plt.legend(loc ='best')
    plt.savefig('../outputs%d/rl/%s-%s-sat=%d.png'%(senario,ylab,xlab,df.sector_sat.iat[0]))
    plt.show()

def ev_curve(df_ev,scenario):
    X=np.array(df_ev.index)
    plt.figure()
    plt.plot(X,df_ev['rsquared'])
    plt.title('Overtime evolution of Goodness of Fit')
    plt.xlabel('time')
    plt.ylabel('R2')
    plt.savefig('../outputs%d/ev/ev_r2.png'%(scenario))
    plt.show()
    plt.figure()
    plt.plot(X,df_ev['alphaEEN'],label='alphaEEN')
    plt.plot(X,df_ev['alpha'],'r-',label='alpha_reg')
    plt.legend(loc='best')
    plt.title('Overtime evolution of wind shear estimated by EEN and regression model')
    plt.xlabel('time')
    plt.savefig('../outputs%d/ev/ev_alpha.png'%(scenario))
    plt.show()
    
def ev_boxplot(df_ev,scenario):
    df_alphas=df_ev[['alphaEEN','alpha']]
    df_alphas.columns=['alphaEEN','alpha_reg']
    df_alphas.boxplot(showmeans=True)
    plt.savefig('../outputs%d/ev/evbox_alpha.png'%(scenario))

def ev_histo(df_ev,scenario):
    discret_r2=df_ev['rsquared']//0.1*0.1
    discret_r2.hist(bins=list(np.arange(0,1.1,0.1)),density=True,label='R2')
    plt.legend(loc='best')
    plt.savefig('../outputs%d/ev/evhist_r2.png'%(scenario))
    plt.show()

def ev_fit(X,Y_real,Y_fitted,Y_EEN,Y_l,Y_u,height_wanted,time,i,log,scenario):
    plt.figure()
    plt.scatter(X,Y_real,label='Real wind data')
    plt.plot(X,Y_fitted,'r-',label='Fitted model')
    plt.scatter(height_wanted,Y_EEN,color='green',label='Prediction by EEN')
    plt.plot(X,Y_l,'r--')
    plt.plot(X,Y_u,'r--',label='IC Prediction')
    plt.legend(loc='best')
    plt.title('Real data vs Fitted model, at %s'%(time))
    if log:
        plt.ylabel('log(v (m/s))')
        plt.xlabel('log(Height (m))')
        plt.savefig('../outputs%d/ev/log_evfit%d.png'%(scenario,i))
    else:
        plt.ylabel('v (m/s)')
        plt.xlabel('Height (m)')
        plt.savefig('../outputs%d/ev/evfit%d.png'%(scenario,i))
    plt.show()

def ev_residu(X,Y_res,time,i,scenario):
    plt.figure()
    plt.plot(X,Y_res,label='Residuals')
    plt.legend(loc='best')
    plt.xlabel('log(Height (m))')
    plt.ylabel('log(v (m/s))')
    plt.title('Variety of residuals, at %s'%(time))
    plt.savefig('../outputs%d/ev/ev_res%d.png'%(scenario,i))

def ev_qqplot(resid_norm,time,i,scenario):
    fig=sm.qqplot(resid_norm,line='45')
    plt.title('Q-Q plot, at %s'%(time))
    plt.savefig('../outputs%d/ev/ev_qqplot%d.png'%(scenario,i))

def rl_fit(X,Y_real,Y_fitted,Y_l,Y_u,sector,speed,cube,scenario):
    plt.figure()
    plt.scatter(X,Y_real,label='Real wind data')
    plt.plot(X,Y_fitted,'r-',label='Fitted model')
    plt.plot(X,Y_l,'y--')
    plt.plot(X,Y_u,'y--',label='IC Prediction')
    plt.legend(loc='best')
    plt.title('Real data vs Fitted model, as sector = %d'%(sector))
    if speed:
        if not cube:
            plt.xlabel('v_sat (m/s)')
            plt.ylabel('v_sit (m/s)')
            plt.savefig('../outputs%d/rlfit_v_%d.png'%(scenario,sector))
        else:
            plt.xlabel('(v_sat)^3 (m/s)^3')
            plt.ylabel('(v_sit)^3 (m/s)^3')
            plt.savefig('../outputs%d/rlfit_v3_%d.png'%(scenario,sector))
    else:
        plt.xlabel('v_sat (m/s)')
        plt.ylabel('Δθ (°)')
        plt.savefig('../outputs%d/rlfit_veer_%d.png'%(scenario,sector))

def rl_residu(X,Y_res,sector,speed,cube,scenario,thres_v,model):
    plt.figure()
    plt.scatter(X,Y_res,label='Residuals')
    plt.legend(loc='best')
    plt.title('Vatriety of residuals, as sector = %d'%(sector))
    if speed:
        if not cube:
            plt.xlabel('v_sat (m/s)')
            plt.ylabel('v (m/s)')
            plt.savefig('../outputs%d/rl/rl%d/rl_res_v_%d_%s.png'%(scenario,thres_v,sector,model))
        else:
            plt.xlabel('(v_sat)^3 (m/s)')
            plt.ylabel('v^3 (m/s)')
            plt.savefig('../outputs%d/rl/rl%d/rl_res_v3_%d_%s.png'%(scenario,thres_v,sector,model))
    else:
        plt.xlabel('v_sat (m/s)')
        plt.ylabel('θ (°)')
        plt.savefig('../outputs%d/rl/rl%d/rl_res_veer_%d_%s.png'%(scenario,thres_v,sector,model))

def rl_qqplot(resid_norm,sector,speed,cube,scenario):
    fig=sm.qqplot(resid_norm,line='45')
    plt.title('Q-Q plot, as sector = %s'%(sector))
    if speed:
        if not cube:
            plt.savefig('../outputs%d/rl_qqplot_v_%d.png'%(scenario,sector))
        else:
            plt.savefig('../outputs%d/rl_qqplot_v3_%d.png'%(scenario,sector))
    else:
        plt.savefig('../outputs%d/rl_qqplot_veer_%d.png'%(scenario,sector))