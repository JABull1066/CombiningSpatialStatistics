import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
from utilities import *
from scipy.io import loadmat
import pandas as pd
import scipy.stats as ss

sns.set(style="ticks")
columnsToCompare = ['Density','maxPCF','maxSCDr','minJ']
n = len(columnsToCompare)
rangeMaxes = [1300,10,1,1]
size = 30

#%% Read in synthetic data and change column names for consistency with other scripts
df_synthetic = pd.read_csv('DataFiles/SyntheticData_TrainingDataset.csv')
df_synthetic.rename(columns = {'True Density per mm squared':'Density'}, inplace = True) 

# Discretise synthetic data according to value of rho
colIncrement = 0.1 # Visualise increments of rho 0.1
targetColorBins = np.round(np.arange(0,1+colIncrement,colIncrement),1)
rho = np.asarray(df_synthetic["Rho"])
cols = np.zeros(len(rho))
for v in range(1,len(targetColorBins)):
    cols[(rho>=targetColorBins[v-1]) & (rho<targetColorBins[v])] = targetColorBins[v]
dataset_withColors = df_synthetic
dataset_withColors['$\\rho$'] = [str(round(v-colIncrement,1)) + ' - ' + str(round(v,1)) for v in list(cols)]
hueOrder = [str(round(v-colIncrement,1)) + ' - ' + str(round(v,1)) for v in targetColorBins[1:]]

        
        
#%% Filtered Head and neck, coloured according to excluded regions
# Read in real datasets, filter to get head and neck data
df = pd.read_csv('DataFiles/CollatedRealDatasets.csv')
df_headAndNeck = df[df.dataset == 'HeadAndNeck']

# Set exclusion thresholds
HEAD_AND_NECK_ANNOTATION_EXCLUSION_THRESHOLD = 0.5 # i.e., 50%
SYNTHETIC_DATA_RHO_EXCLUSION_THRESHOLD = 0.5

# Visualise colours in the right order, set name of labels
hueOrder = [str(int((v-colIncrement)*100)) + '% - ' + str(int(v*100)) + '%' for v in targetColorBins[1:]]
hueOrder = hueOrder[0:5]

# Filter the data
df_headAndNeck_filtered = df_headAndNeck[df_headAndNeck['ROI_propAnnotatedOut'] < HEAD_AND_NECK_ANNOTATION_EXCLUSION_THRESHOLD]
df_synthetic_filtered = dataset_withColors[dataset_withColors['Rho'] <= SYNTHETIC_DATA_RHO_EXCLUSION_THRESHOLD]

#%% We need to combine datasets together, because some pillock wasn't consistent with how they named the column headers (sorry)
comparisonDataset = []
for i in range(len(df_headAndNeck_filtered)):
    data = ['Head and Neck','Head and Neck']
    for v in columnsToCompare:
        data.append(df_headAndNeck_filtered[v].iloc[i])
    comparisonDataset.append(data)
    
columnsToCompare.append('Rho')
        
for i in range(len(df_synthetic_filtered)):
    data = ['Synthetic',df_synthetic_filtered['$\\rho$'].iloc[i]]
    for v in columnsToCompare:
        data.append(df_synthetic_filtered[v].iloc[i])
    comparisonDataset.append(data)


columnNames = ['Dataset','$\\rho$',]
columnNames.extend(columnsToCompare)
df_comparison_filtered = pd.DataFrame(comparisonDataset, columns=columnNames) 

#%% OK, now plot Figure 3a
sns.set(style='white',font_scale=2)
hueOrder = ['0.0 - 0.1','0.1 - 0.2','0.2 - 0.3','0.3 - 0.4','0.4 - 0.5','Head and Neck']
pal = sns.color_palette('plasma',len(hueOrder)-1)
pal.append((0,0,0))
markers = ['o','o','o','o','o','o']
df_comparison_filtered = df_comparison_filtered.rename({"maxPCF":"$g_{max}$","maxSCDr":"$F_{max}$","minJ":"$J_{min}$"}, axis='columns')
columnsToCompare = ["Density","$g_{max}$","$F_{max}$","$J_{min}$"]

df_comparison_filtered_crop = df_comparison_filtered
g = sns.PairGrid(df_comparison_filtered_crop,vars=columnsToCompare,hue='$\\rho$',hue_order=hueOrder,palette=pal,diag_sharey=False)


g.map_diag(sns.kdeplot, lw=2, legend=False,shade=False)
g.map_lower(plt.scatter,s=5)


for i in range(n):
    for j in range(n):
        g.axes[i,j].set_xlim([0,rangeMaxes[j]])
        g.axes[i,j].set_ylim([0,rangeMaxes[i]])
        if i < j:
            g.axes[i, j].set_visible(False)
        if i == 0:
            g.axes[i,j].set_yticklabels('')
            g.axes[i,j].set_ylabel('')

g.fig.subplots_adjust(hspace=0.2,wspace=0.25)