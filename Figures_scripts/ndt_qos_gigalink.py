# -*- coding: utf-8 -*-
"""
NDT Dataset - Changepoint Detection - QoS
@author: Cleiton Moya de Almeida
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

clients = ['dca6326b9aa1', 'dca6326b9ada', 'dca6326b9b52', 'dca6326b9c99',
 'dca6326b9ca8', 'dca6326b9ce4', 'e45f011e2d20', 'e45f0134230d',
 'e45f01359a20', 'e45f013607cb', 'e45f0136103e', 'e45f013610c8',
 'e45f018e51b7', 'e45f018e5242', 'e45f01963bb8', 'e45f01963c21',
 'e45f01ad569d', 'e45f01b4bb1e', 'e45f01b4bbc1']


clients2 = [ 'e45f011e2d20', 'e45f0134230d', 'e45f018e51b7']
#clients2 = [ 'e45f011e2d20', 'e45f0134230d', 'e45f018e51b7', 'e45f01ad569d']

client_n = [f'Client {n}' for n in range(1,len(clients)+1)]
dict_client = {c:n+1 for n,c in enumerate(clients)}

series = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']

sites = ['gig01', 'gig02', 'gig03', 'gig04', 
         'gru02', 'gru03', 'gru05','rnp_rj', 'rnp_sp']

methods = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'vwcd', 'pelt_np']

address = {
    'e45f011e2d20':'Nova Suíça', 
    'e45f0134230d':'Olaria', 
    'e45f018e51b7':'Riograndina', 
    'e45f01ad569d':'Data center',
    }

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


lw = 0.5 # linewidth to plot

# Formatters
markers_dict = {
    'Shewhart' :'x', 
    'EWMA': 's', 
    '2S-Cusum': '^', 
    'WL-Cusum': 'v',
    'VWCD': 'd',
    'Pelt-NP': '*'}

color_dict = {
    'Shewhart' :'r', 
    'EWMA': 'r', 
    '2S-Cusum': 'r', 
    'WL-Cusum': 'r',
    'VWCD': 'b',
    'Pelt-NP': 'g'}

size_dict = {
    'Shewhart' :3, 
    'EWMA': 3, 
    '2S-Cusum': 3, 
    'WL-Cusum': 3,
    'VWCD': 4,
    'Pelt-NP': 5}



# Load the dataframe of results
df = pd.concat([pd.read_pickle(f'../Experiment/results_ndt/df_ndt_{m}.pkl') 
                for m in methods], ignore_index=True)

df['method_name'] = [method_name(m) for m in df['method']]


# Plot the figure
#fig = plt.figure(constrained_layout=True, figsize=(5,5))
#ax = fig.subplot_mosaic([['legend'],[0], [1], [2], [3]], 
#                          gridspec_kw={'height_ratios':[0.001, 1, 1, 1, 1]},
#                          sharex=True)

fig = plt.figure(constrained_layout=True, figsize=(5,4))
ax = fig.subplot_mosaic([['legend'],[0], [1], [2]], 
                          gridspec_kw={'height_ratios':[0.001, 1, 1, 1]},
                          sharex=True)

ax['legend'].axis('off')

for j,client in enumerate(clients2):
    
    # Load the timeseries
    site = 'rnp_rj'
    serie = 'd_throughput'
    file = f'{client}_{site}_{serie}.txt'
    df_s = pd.read_csv(f'../Dataset/ndt/{file}', parse_dates=True, names=['t', 'y'], index_col=0)
    #y = np.loadtxt(f'../Dataset/ndt/{file}', delimiter=',')
    y = df_s.y.values
    t = df_s.index.values
    
    # Load the list of changepoints (not empty) for each method
    df_ = df[(df.client==client) &
             (df.site==site) &
             (df.serie==serie) &
             (df.CP.str.len() != 0)]

    # dict of methods chagepoint list
    CP_dict = dict(df_[['method', 'CP']].to_dict(orient='split')['data'])

    methods_list = df_[df_.method.isin(methods)].method.unique()
    #methods_set.add(methods_list)

    ax['legend'].axis('off')
    ax[j].set_title(f'Client {dict_client[client]} ({address[client]})')
    ax[j].plot(t, y, linewidth=lw)
    ax[j].grid(linestyle=':')
    ax[j].set_ylabel('Mbits/s', fontsize=6)
    ax[j].tick_params(axis='both', labelsize=6)
    
    y0,y1 = ax[j].get_ylim()
    y2=0.8*y1
    d = (y2-y0)/5
    y0 = y0+d

    for i,m in enumerate(methods_list):
        CP = CP_dict[m]
        for cp in CP:
            ax[j].axvline(t[cp], 
                          color = color_dict[method_name(m)],
                          linestyle='-', 
                          alpha=0.5, 
                          linewidth=0.5)
            
            ax[j].plot(t[cp], y0+i*d, 
                       marker = markers_dict[method_name(m)], 
                       markersize = size_dict[method_name(m)], 
                       color = color_dict[method_name(m)])

    #ax[j].xaxis.set_tick_params(labelbottom=True)
    ax[j].set_ylim([0,1000])
    ax[j].fill_between(t,0,1000, where = t>pd.to_datetime('2024-07-04T16:00:00'), 
                       color='gray', alpha=0.15)
ax[0].set_xlim(left=pd.to_datetime('2024-06-20'), right=pd.to_datetime('2024-07-10'))


locator = mdates.AutoDateLocator(minticks=1, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax[0].xaxis.set_major_locator(locator)
ax[0].xaxis.set_major_formatter(formatter)
ax[0].set_xticks([19895, 19899, 19903, 19905, 19906, 19907,19908, 19909, 
                  19910, 19911, 19912, 19913])


# draw the legend
lines_leg = [mlines.Line2D([], [], 
                           color = color_dict[method_name(m)], 
                           marker = markers_dict[method_name(m)], 
                           linewidth=0, 
                           markersize = size_dict[method_name(m)], 
                           label=method_name(m)) for m in methods]


event_leg = mpatches.Patch(color='gray', label='Known event', alpha=0.3)

lines_leg.append(event_leg)

_ = ax['legend'].legend(handles=lines_leg,
                    loc='upper center',
                    ncol=3, 
                    fontsize=6,
                    handletextpad=0.3,
                    columnspacing=0.5)