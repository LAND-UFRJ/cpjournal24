# -*- coding: utf-8 -*-
"""
Shao Experiment - Confusion Matrix
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shao_benchmark import evaluation_window
import seaborn as sns

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

methods = ['ewma_ps', 'vwcd', 'pelt_np']

df = pd.concat([pd.read_pickle(f'../Experiment/results_shao/df_shao_{m}.pkl') for m in methods], 
               ignore_index=True)

series1 = df.serie.tolist()
series = [s[:-1] for s in series1]
path = '../Dataset/shao/'
we = 5

# Read the series lenght
N = []
for s in series:
    y = np.loadtxt(f'{path}{s}.csv', usecols=1, delimiter=';', skiprows=1)
    N.append(len(y))

dict_N = {s:n for s,n in zip(series1,N)}

Tp = []
Fp = []
Tn = []
Fn = []
for _,r in df.iterrows():
    n = dict_N[r.serie]
    CP_label = r.CP_label
    CP_pred = r.CP_pred
    metrics = evaluation_window(CP_label, CP_pred, window=we)
    
    tp = metrics['tp']
    fp = metrics['fp']
    fn = metrics['fn']
    tn = n-tp-fp-fn
    
    Tp.append(tp)
    Fp.append(fp)
    Fn.append(fn)
    Tn.append(tn)

df['tp'] = Tp
df['tn'] = Tn
df['fp'] = Fp
df['fn'] = Fn

# Aggregate the metrics (sum)
df_ = df[['method', 'tn', 'fp', 'fn', 'tp']].groupby('method').sum()

#cm_bocd = a = df_.loc['bocd_ps'].to_numpy().reshape(2,2)
cm_vwcd = a = df_.loc['vwcd'].to_numpy().reshape(2,2)
cm_pelt = a = df_.loc['pelt_np'].to_numpy().reshape(2,2)
cm_ewma = a = df_.loc['ewma_ps'].to_numpy().reshape(2,2)
#Titles = ["BOCD (proposed)", "VWCD", "Pelt-NP"]
Titles = ["EWMA (proposed)", "VWCD", "Pelt-NP"]
cms = np.array([cm_ewma, cm_vwcd, cm_pelt])


cm_vwcd_nor = cm_vwcd / cm_vwcd.sum(axis=1)[:, np.newaxis]
cm_pelt_nor = cm_pelt / cm_pelt.sum(axis=1)[:, np.newaxis]
cms_nor = np.array([cm_vwcd_nor, cm_pelt_nor])

fig,ax = plt.subplots(figsize=(3.5,1.5), ncols=3, sharey=True, sharex=True, 
                     layout='constrained')

vmin = cms.flatten().min()
vmax = cms.flatten().max()
vmax = 500000
for i,cm in enumerate(cms): 


    off_diag_mask = np.eye(*cm.shape, dtype=bool)    

    if i==0:
        labels = np.array([f'{t}\n{l}' for t,l in zip(cm.flatten(),['(TN)','(FP)','(FN)','(TP)'])]).reshape(2,2)
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=labels, fmt='', square=True,
                    linewidths=0.5, linecolor='k', 
                    #cmap='gray_r', vmin=0, vmax=0)  # set all to white)
                    cmap='Blues', mask=~off_diag_mask, vmin=vmin, vmax=vmax)
        
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=labels, fmt='', square=True,
                    linewidths=0.5, linecolor='k',
                    #cmap='gray_r', vmin=0, vmax=0)  # set all to white)
                    mask=off_diag_mask, cmap='Reds', vmin=vmin, vmax=vmax)

    else:
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=True, fmt='d', square=True,
                    linewidths=0.5, linecolor='k',
                    cmap='Blues', mask=~off_diag_mask, vmin=vmin, vmax=vmax)
        
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=True, fmt='d', square=True,
                    linewidths=0.5, linecolor='k',
                    mask=off_diag_mask, cmap='Reds', vmin=vmin, vmax=vmax)

    ax[i].set_title(Titles[i])
    sns.despine(left=False, right=False, top=False, bottom=False)

ax[0].set_xlabel('Predicted label')
ax[0].set_ylabel('True label')

# Normalized
fig,ax = plt.subplots(figsize=(2.5,1.5), ncols=2, sharey=True, sharex=True, 
                     layout='constrained')
vmin = 0
vmax = 1
for i,cm in enumerate(cms_nor): 


    off_diag_mask = np.eye(*cm.shape, dtype=bool)    

    if i==0:
        labels_ = np.array([[f'{cm[0,0]:.4f}', f'{cm[0,1]:.4f}'], [f'{cm[1,0]:.2f}', f'{cm[1,1]:.2f}']])
        labels = np.array([f'{s}\n{l}' for s,l in zip(labels_.flatten(),['(TNR)','(FPR)','(FNR)','(TPR)'])]).reshape(2,2)
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=labels, fmt='', square=True,
                    linewidths=0.5, linecolor='k', 
                    #cmap='gray_r', vmin=0, vmax=0)  # set all to white)
                    cmap='Blues', mask=~off_diag_mask, vmin=vmin, vmax=vmax)
        
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=labels, fmt='', square=True,
                    linewidths=0.5, linecolor='k',
                    #cmap='gray_r', vmin=0, vmax=0)  # set all to white)
                    mask=off_diag_mask, cmap='Reds', vmin=vmin, vmax=vmax)

    else:
        labels = np.array([[f'{cm[0,0]:.4f}', f'{cm[0,1]:.4f}'], [f'{cm[1,0]:.2f}', f'{cm[1,1]:.2f}']])
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=labels, fmt='', square=True,
                    linewidths=0.5, linecolor='k',
                    cmap='Blues', mask=~off_diag_mask, vmin=vmin, vmax=vmax)
        
        sns.heatmap(cm, ax=ax[i], cbar=False, annot=labels, fmt='', square=True,
                    linewidths=0.5, linecolor='k',
                    mask=off_diag_mask, cmap='Reds', vmin=vmin, vmax=vmax)

    ax[i].set_title(Titles[i])
    sns.despine(left=False, right=False, top=False, bottom=False)

ax[0].set_xlabel('Predicted label')
ax[0].set_ylabel('True label')