# -*- coding: utf-8 -*-
"""
NDT Dataset - Results 
- Boxplot for the number of change-points and elapsed-time
- QoS application example - Download quality worsening
- 
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']
series_name = ['Download throughput', 'Download RTT ', 'Upload throughput', 'Upload RTT']

clients = ['dca6326b9aa1', 'dca6326b9ada', 'dca6326b9b52', 'dca6326b9c99',
 'dca6326b9ca8', 'dca6326b9ce4', 'e45f011e2d20', 'e45f0134230d',
 'e45f01359a20', 'e45f013607cb', 'e45f0136103e', 'e45f013610c8',
 'e45f018e51b7', 'e45f018e5242', 'e45f01963bb8', 'e45f01963c21',
 'e45f01ad569d', 'e45f01b4bb1e', 'e45f01b4bbc1']

clients2 = ['dca6326b9aa1', 'dca6326b9c99', 'dca6326b9ca8',
       'dca6326b9ce4', 'e45f01359a20', 'e45f01963c21']

clients_giga = ['dca6326b9b52', 'e45f011e2d20', 'e45f0134230d', 'e45f013607cb',
    'e45f0136103e', 'e45f018e51b7', 'e45f018e5242', 'e45f01963bb8',
    'e45f01ad569d', 'e45f01b4bb1e', 'e45f01b4bbc1'] 


dict_client = {c:n+1 for n,c in enumerate(clients)}

sites = ['gig01', 'gig02', 'gig03', 'gig04',
         'gru02', 'gru03', 'gru05', 'gru06', 'fln01']

methods_name = ['Shewhart', 'EWMA', '2S-CUSUM', 'WL-CUSUM', 'VWCD', 'Pelt-NP']
sequential_ps = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps']
methods_param = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps', 'vwcd']
hatches = ['', '////']

methods = ['shewhart_ba', 'shewhart_ps', 'ewma_ba', 'ewma_ps',
           'cusum_2s_ba', 'cusum_2s_ps', 'cusum_wl_ba', 'cusum_wl_ps',
           'vwcd', 'pelt_np']


# Read the dataset
df = pd.concat([pd.read_pickle(f'../Experiment/results_ndt/df_ndt_{m}.pkl') for m in methods], 
               ignore_index=True)

def dev_abs(M, dev_abs, direction):
    M = np.array(M)
    n_mean = 0
    if len(M) > 1:
        dM = np.diff(M)
        
        if direction == 'dec':
            c2 = dM <= -dev_abs
        else:
            c2 = dM >= dev_abs
        n_mean = c2.sum()
    return n_mean

def method_type(m):
    if m[-2:] == 'ba':
        return 'Basic'
    elif m=='pelt_np':
        return 'Reference (off-line)'
    else:
        return 'Proposed'

def method_name(m):
    if m[:8] == 'shewhart':
        return 'Shewhart'
    elif m[:4] == 'ewma':
        return 'EWMA'
    elif m[:8] == 'cusum_2s':
        return '2S-Cusum'
    elif m[:8] == 'cusum_wl':
        return 'WL-Cusum'
    elif m=='vwcd':
        return 'VWCD'
    elif m[:4]=='bocd':
        return 'BOCD'
    elif m=='rrcf_ps':
        return 'RRCF'
    elif m=='pelt_np':
        return 'Pelt-NP'


def method_order(m):
    if m[:8] == 'shewhart':
        return 0
    elif m[:4] == 'ewma':
        return 1
    elif m[:8] == 'cusum_2s':
        return 2
    elif m[:8] == 'cusum_wl':
        return 3
    elif m[:4]=='bocd':
        return 4
    elif m=='rrcf_ps':
        return 5
    elif m=='vwcd':
        return 6
    elif m=='pelt_np':
        return 7


flierprops = dict( markersize=2)
hatches = ['', '///', '-', 'x', '\\', '*', 'o', 'O', '.']
C0 = np.array([142, 186, 217])/255 # blue
C1 = np.array([255, 190, 134])/255 # orange
C2 = np.array([149, 207, 149])/255 # green

# Download quality worsening in the mean per client and serie - function of delta
markers = [',', 'o', 'x', 'v', 's', 'D']
methods_name = ['Shewhart', 'EWMA', '2S-CUSUM', 'WL-CUSUM', 'VWCD']
k=0

fig = plt.figure(constrained_layout=True, figsize=(5,3.7))
ax = fig.subplot_mosaic([['legend', 'legend', 'legend', 'legend'],[0,1,2,3], [4,5,6,7], [8,9,10,11]], 
                          gridspec_kw={'height_ratios':[0.001, 1, 1, 1]})

ax['legend'].axis('off')
xlim = 200
ylim = 20
x_thr = np.arange(1, xlim+1, 1)
sites2 = ['gru03']

for i,c in enumerate(clients_giga):
    ax[i].set_title(f'Client {dict_client[c]} ({c})', fontsize=8)
    ax[i].grid(linestyle=':')
    
    for j,m in enumerate(methods_param):
        
        df_ = df[(df.client==c) & (df.serie==series_type[k]) & (df.method==m)
                 & (df.site.isin(sites2))]
        
        M0 = df_.M0.values.tolist()
        M = [sum([dev_abs(m0_list, p, 'dec') 
              for m0_list in M0]) for p in x_thr]
        ax[i].plot(x_thr, M, label=f'{methods_name[j]}')
        ax[i].set_yticks(range(0,ylim+2,2))
        ax[i].set_yticklabels(range(0,ylim+2,2))
        ax[i].set_xticks(range(0,250,50))
        ax[i].set_xticklabels(range(0,250,50))
        ax[i].set_ylim([0,ylim])
        ax[i].set_xlim([0,xlim])
        ax[i].tick_params(axis='both', which='major', labelsize=8)   
        
        if i==0:
            ax[i].set_ylabel('Num. of changepoints', fontsize=8)
            ax[i].set_xlabel('Decrement (Mbits/s)', fontsize=8)
            
handles, labels = ax[0].get_legend_handles_labels()
_ = ax['legend'].legend(handles, labels, loc="upper center", ncol=3, fontsize=8)


#%%
flierprops = dict( markersize=2, markeredgewidth=0.5)
medianprops = dict(linewidth=1)
fig,ax = plt.subplots(figsize=(7,2), nrows=1, ncols=4, sharey=True, layout='constrained')
for i,s in enumerate(series_type):
    df_ = df[(df['method'].isin(sequential_ps)) & (df.serie == s) & (df.client.isin(clients_giga))]
    df_ = df_[['client', 'serie', 'num_anom_u', 'num_anom_l']].melt(id_vars=['client', 'serie'], var_name='anom_type', value_name='num_anom')
    
    ax[i].set_title(series_name[i])
    bar = sns.boxplot(data=df_, x="client", y="num_anom", hue='anom_type', 
                      ax=ax[i], palette = [C0, C1], saturation=1,
                      linewidth=0.5, flierprops=flierprops, zorder=2,
                      medianprops=medianprops)
    
    ax[i].grid(axis='y', linestyle=':')
    
    if i==0:
        bar.legend_.texts[0].set_text('Incremento')
        bar.legend_.texts[1].set_text('Decremento')
        bar.legend_.set_title(None)
        bar.legend_.set_frame_on(True)
        bar.legend_.get_patches()[0].set_hatch(hatches[0])
        bar.legend_.get_patches()[1].set_hatch(hatches[1])
    else:
        ax[i].get_legend().remove()
     
    '''
    for j, bar in enumerate(ax[i].patches):
        if j<12:
            hatch = hatches[j//6]
        else:
            hatch = hatches[j-12]  
        bar.set_hatch(hatch)
    '''
    ax[i].set_xticks(ax[i].get_xticks())
    ax[i].set_xticklabels(range(1,12))
    if i==0:
        ax[i].set_ylabel("Núm. de anomalias pontuais")
        ax[i].set_xlabel("Cliente")
    else:
        ax[i].set_ylabel("")
        ax[i].set_xlabel("")