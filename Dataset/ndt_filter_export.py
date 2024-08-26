# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 23:25:41 2023
Analisa séries coletadas
@author: cleiton
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})


export = True

# Importação dos dados
df = pd.read_pickle('df_med.pkl')

# Filtros
clients = ['dca6326b9aa1',
           'dca6326b9ada',
           'dca6326b9c99',
           'dca6326b9ca8',
           'e45f01359a20',
           'e45f01963c21',
           'dca6326b9ce4',
           'e45f01963bb8',
           'e45f01ad569d',
           'e45f01ad5631', # gigalink
           'e45f018e5242',
           'e45f0134230d',
           'e45f01b4bb1e',
           'e45f018e527a',
           'e45f013607cb',
           'e45f013610c8',
           'e45f018e51b7',
           'e45f0136103e',
           'e45f011e2d20',
           'e45f01b4bbc1',
           'dca6326b9b52']

# Data de início que as leituras estão confiáveis
data = ['2023-05-24',
        '2023-07-06',
        '2023-05-09', 
        '2023-05-13',
        '2023-05-01', #2023-04-28
        '2023-05-12',
        '2023-06-13',
        '2023-10-05',
        '2023-10-18',
        '2023-05-01', #Gigalink
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01',
        '2023-05-01']


# Período considerado
data_final = '2024-07-11 23:59:59'

data_med = {c:d for (c,d) in zip(clients,data)}
sites = df.Site.unique()
series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']

med = []

'''
for c in clients:
    for j,s in enumerate(sites):
        df_d = df[(df.ClientMac == c) & (df.DownloadSite == s) & (df.DataHora >= data_med[c]) & (df.DataHora <= data_final)]
        df_u = df[(df.ClientMac == c) & (df.UploadSite == s) & (df.DataHora >= data_med[c]) & (df.DataHora <= data_final)]
        
        
        if len(df_d) >= 100:
            if export:
                np.savetxt(f'ndt/{c}_{s}_d_rttmean.txt', df_d[['DataHora', 'DownRTTMean']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'ndt/{c}_{s}_d_throughput.txt', df_d[['DataHora', 'Download']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'ndt/{c}_{s}_d_retrans.txt', df_d[['DataHora', 'DownloadRetrans']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'ndt/{c}_{s}_u_rttmean.txt', df_u[['DataHora', 'UpRTTMean']].values, fmt=['%s', '%.3f'], delimiter=',')
                np.savetxt(f'ndt/{c}_{s}_u_throughput.txt', df_u[['DataHora', 'Upload']].values, fmt=['%s', '%.3f'], delimiter=',')
        
            inicio_d = df_d.DataHora.iloc[0]
            fim_d = df_d.DataHora.iloc[-1]
            num_med_d = len(df_d)
            mean_t_d = np.round(df_d.DataHora.diff().mean().seconds/3600,1)
            num_med_u = len(df_u)
            file_prefix = f"{c}_{s}_"
            
            quant = {"client": c, 
                     "site": s, 
                     "inicio": inicio_d,
                     "fim": fim_d,
                     "num_med_d": num_med_d,
                     "num_med_u": num_med_u,
                     "mean_t": mean_t_d,
                     "file_prefix": file_prefix
                     }
            med.append(quant)
            
            # Verifica se não há número de medições de download-upload diferentes
            #if num_med_d != num_med_u:
            #    print(f'Client: {c}, Site: s, num_med_d:{num_med_d}, num_med_u:{num_med_u}')


# Conjunto de medições
df_series = pd.DataFrame(med)
'''

#df_series.to_pickle('df_series.pkl')
df_series = pd.read_pickle('df_series.pkl')
print('Sites com mais de 100 medições:', len(df_series.site.unique()))
print(df_series.site.unique())

# Statistics
print('\nInitial date:', sorted(df_series[df_series.inicio!='-'].inicio)[0])
print('Final date:', sorted(df_series[df_series.inicio!='-'].fim)[-1])
print('Number of clients:', len(df_series.client.unique()))
print('Number of measurements (download):', df_series.num_med_d.sum())
print('Number of measurements (upload):', df_series.num_med_u.sum())
print('Number of séries:', len(df_series)*4)
print('Mean d:', df_series.num_med_d.mean())
print('Min d:', df_series.num_med_d.min())
print('Max d:', df_series.num_med_d.max())
print('Mean time between observations (down):', df_series.mean_t.mean())