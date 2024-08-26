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

clients_giga2 = ['dca6326b9b52', 'e45f011e2d20', 'e45f0134230d', 'e45f01ad569d',
    'e45f0136103e', 'e45f01963bb8'] 

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


medias = df.M0.values.tolist()
DM = []
for m in medias:
    if m is not None:
        if len(m) > 1:
            dM = np.diff(m).tolist()
            DM.append(dM)
        else:
            DM.append([])
    else:
        DM.append(None)
df['dm'] = DM

df_ = df[(df.method=='vwcd') & (df.serie=='d_throughput') & (df.site=='rnp_rj') 
         & (df.client.isin(clients_giga))]

client_ = df_.client.values.tolist()
dm_ = df_.dm.values.tolist()

dict_c = {c:d for c,d in zip(client_,dm_)}
