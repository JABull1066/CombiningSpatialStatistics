import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5)
from utilities import *
import scipy.stats as ss
from scipy.optimize import curve_fit
sns.set_style('white')
import matplotlib.cm as cm
import matplotlib.colors as colors

def func_negExponential(x, a, b,c):
    # Helper function for plotting a negative exponential curve
    return a*np.exp(-b*x) + c

noiseParam = '$\\rho$'
predictedNoiseParam = '$\\eta$'

    
# Axes limits for plotting
glim = [0,4]
flim = [0,1.4]
jlim = [-0.02,1.02]


# Restrict density range of data points
densityRange = [150,10000]
# Restrict data to GRF with proportion of tumour area within this range (vs stroma)
tumourPropRange = [0.25,0.75]
# Restrict range of rho considered
MASTER_RHO_THRESHOLD = 0.5

#%% First, we see how the spatial statistics vary with eta

# Training data file:
inputFile = 'DataFiles/SyntheticData_TrainingDataset.csv'
df_train = pd.read_csv(inputFile)
df_train = df_train[df_train['Rho'] <= MASTER_RHO_THRESHOLD]
df_train = df_train[(df_train['True Density per mm squared'] >= densityRange[0]) & (df_train['True Density per mm squared'] <= densityRange[1])]
df_train = df_train[(df_train['Tumour Area (mm2)']/2.25 >= tumourPropRange[0]) & (df_train['Tumour Area (mm2)']/2.25 <= tumourPropRange[1])]

d_rho = 0.02
rhos = np.round(np.arange(0,MASTER_RHO_THRESHOLD+d_rho,d_rho),2)

g_grid = np.arange(1,5,0.025)
F_grid = np.arange(0,1.5,0.025)
J_grid = np.arange(0,1,0.01)



#%% Make Figure 3 (b-d)

maxPCF_mean = []
maxPCF_SD = []
maxSCDr_mean = []
maxSCDr_SD = []
minJ_mean = []
minJ_SD = []

maxPCF_mean_normfit = []
maxPCF_SD_normfit = []

binCounts = []
for rho in rhos:
    rho = np.round(rho,decimals=2)
    results = df_train[np.round(df_train['Rho'],decimals=2)==rho]

    minJ_mean.append(np.mean(results['minJ']))
    maxSCDr_mean.append(np.mean(results['maxSCDr']))
    
    minJ_SD.append(np.std(results['minJ']))
    maxSCDr_SD.append(np.std(results['maxSCDr']))
    
    # For PCF, discard extreme outliers
    results = results[results['maxPCF'] < 10]
    
    maxPCF_mean.append(np.mean(results['maxPCF']))
    maxPCF_SD.append(np.std(results['maxPCF']))
    


maxPCF_mean = np.asarray(maxPCF_mean)
maxPCF_SD = np.asarray(maxPCF_SD)
maxSCDr_mean = np.asarray(maxSCDr_mean)
maxSCDr_SD = np.asarray(maxSCDr_SD)
minJ_mean = np.asarray(minJ_mean)
minJ_SD = np.asarray(minJ_SD)

# Store coefficients of exponentials for later use
meanCoefficients = []
SDCoefficients = []

markerSize = 10
errorgap = 2
plt.figure(figsize=(9,9))
plt.errorbar(rhos,maxPCF_mean,yerr=maxPCF_SD,fmt='.k',ms=markerSize,errorevery=errorgap)
plt.ylabel('$g_{max}$')
plt.xlabel(noiseParam)
plt.ylim([0,3.7])
sns.despine()
x = rhos
y = maxPCF_mean
params = curve_fit(func_negExponential, x, y,p0=[-1,1,1],sigma=maxPCF_SD)
[a, b, c] = params[0]
plt.plot(rhos,func_negExponential(rhos,a,b,c),c='k',linestyle='--',label='Exponential fit')
print([a,b,c])
meanCoefficients.append([a,b,c])

plt.figure(figsize=(9,9))
plt.errorbar(rhos,maxSCDr_mean,yerr=maxSCDr_SD,fmt='.k',ms=markerSize,errorevery=errorgap)
plt.ylabel('$F_{max}$')
plt.xlabel(noiseParam)
plt.ylim([0,1.5])
sns.despine()
y = maxSCDr_mean
params = curve_fit(func_negExponential, x, y,sigma=maxSCDr_SD,p0=[1,10,0])
[a, b, c] = params[0]
plt.plot(rhos,func_negExponential(rhos,a,b,c),c='k',linestyle='--',label='Exponential fit')
print([a,b,c])
meanCoefficients.append([a,b,c])

plt.figure(figsize=(9,9))
plt.errorbar(rhos,minJ_mean,yerr=minJ_SD,fmt='.k',ms=markerSize,errorevery=errorgap)
plt.ylabel('$J_{min}$')
plt.xlabel(noiseParam)
plt.ylim([0,1])
sns.despine()
x = rhos
y = minJ_mean
params = curve_fit(func_negExponential, x, y,sigma=minJ_SD,p0=[-1,1,1])
[a, b, c] = params[0]
plt.plot(rhos,func_negExponential(rhos,a,b,c),c='k',linestyle='--',label='Exponential fit')
print([a,b,c])
meanCoefficients.append([a,b,c])



# Compare the SDs
plt.figure(figsize=(9,9))
plt.scatter(rhos,maxPCF_SD,c='b',label='$g_{max}$')
params = curve_fit(func_negExponential, rhos, maxPCF_SD)
[a, b, c] = params[0]
SDCoefficients.append([a,b,c])
plt.plot(rhos,func_negExponential(rhos,a,b,c),c='b',linestyle='--',label='Exponential fit')


plt.scatter(rhos,maxSCDr_SD,c='g',label='$F_{max}$')
params = curve_fit(func_negExponential, rhos, maxSCDr_SD,p0=[1,10,0])
[a, b, c] = params[0]
SDCoefficients.append([a,b,c])
plt.plot(rhos,func_negExponential(rhos,a,b,c),c='g',linestyle='--',label='Exponential fit')


plt.scatter(rhos,minJ_SD,c='r',label='$J_{min}$')
params = curve_fit(func_negExponential, rhos, minJ_SD,p0=[-1,1,1])
[a, b, c] = params[0]
SDCoefficients.append([a,b,c])
plt.plot(rhos,func_negExponential(rhos,a,b,c),c='r',linestyle='--',label='Exponential fit')

plt.xlabel(noiseParam)
plt.ylabel('Standard deviation')
plt.legend()
sns.despine()

    
    
    
#%% Make Figure 4 components
# Predict etas for new data
trainingRho = [df_train['Rho']]
trainingStats = [df_train['maxPCF'],df_train['maxSCDr'],df_train['minJ']]


g_grid = np.arange(1,5,0.01)
F_grid = np.arange(0,1.5,0.01)
J_grid = np.arange(0,1,0.01)
rangesToGenerateTable = [g_grid, F_grid, J_grid]

# Testing data file
inputFile = 'DataFiles/SyntheticData_TestingDataset.csv'

df_test = pd.read_csv(inputFile)
df_test = df_test[df_test['Rho'] <= MASTER_RHO_THRESHOLD]
df_test = df_test[(df_test['True Density per mm squared'] >= densityRange[0]) & (df_test['True Density per mm squared'] <= densityRange[1])]
df_test = df_test[(df_test['Tumour Area (mm2)']/2.25 >= tumourPropRange[0]) & (df_test['Tumour Area (mm2)']/2.25 <= tumourPropRange[1])]


iteration = df_test['Iteration']
lengthscale = df_test['Length Scale (mm)']
minJ = df_test['minJ'] 
maxSCDr = df_test['maxSCDr'] 
maxPCF = df_test['maxPCF'] 
density_unclipped = df_test['True Density per mm squared']
trueEtas_unclipped = df_test['Rho']
tumourProp_unclipped = df_test['Tumour Area (mm2)']/2.25


testSets = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
plotLoc = [[0,0],[1,1],[2,2],[1,0],[2,0],[2,1],[0,2]]

names = ['$g_{max}$','$F_{max}$','$J_{min}$']#,'$d$']
observations = [maxPCF,maxSCDr,minJ]#,density]


CIwidths = []
for indInd in range(len(testSets)):
    trueEtas = trueEtas_unclipped
    density = density_unclipped
    tumourProp = tumourProp_unclipped
    indices = testSets[indInd]
    
    
    r = [rangesToGenerateTable[v] for v in indices]
    means = [meanCoefficients[v] for v in indices]
    sds = [SDCoefficients[v] for v in indices]
    trainingData = [trainingStats[v] for v in indices]
    trainingData = np.concatenate((trainingRho,trainingData),axis=0)
    
    etaValues, logLikelihoodTable = GenerateLogLikelihoodTable_FittedMeanAndSD(trainingData, r, means, sds)


    predictedEtas = []
    CIs = []
    n = len(indices)
    for i in range(len(df_test)):
        indList = []
        for index in indices:
            indList.append(find_nearest_index(rangesToGenerateTable[index], observations[index].iloc[i]))
            
        if n == 1:
            predictedEtas.append(etaValues[np.argmax(logLikelihoodTable[:,indList[0]])])
            profileLogLikelihood = logLikelihoodTable[:,indList[0]]
        if n == 2:
            predictedEtas.append(etaValues[np.argmax(logLikelihoodTable[:,indList[0],indList[1]])])
            profileLogLikelihood = logLikelihoodTable[:,indList[0],indList[1]]
        if n == 3:
            predictedEtas.append(etaValues[np.argmax(logLikelihoodTable[:,indList[0],indList[1],indList[2]])])
            profileLogLikelihood = logLikelihoodTable[:,indList[0],indList[1],indList[2]]
        if n == 4:
            predictedEtas.append(etaValues[np.argmax(logLikelihoodTable[:,indList[0],indList[1],indList[2],indList[3]])])
            profileLogLikelihood = logLikelihoodTable[:,indList[0],indList[1],indList[2],indList[3]]
        
        thresh = 0.5*ss.chi2.ppf([0.95], df=1)[0]
        maxLike = max(profileLogLikelihood)
        confInterval = profileLogLikelihood > maxLike-thresh
        if len(np.where(confInterval)[0])>0:
            CI = [etaValues[min(np.where(confInterval)[0])],etaValues[max(np.where(confInterval)[0])]]
        else:
            CI = [np.nan,np.nan]
        CIs.append(CI)

    predictedEtas = np.asarray(predictedEtas)
    CIs = np.asarray(CIs)
    
    sns.set(style='white',font_scale=2)

    print(indices)
    titl = ', '.join([names[v] for v in indices])
    eta_subscript = ', '.join([names[v][1] for v in indices])
    
    
    predictedEtas = np.asarray(predictedEtas[(density >= densityRange[0]) & (density <= densityRange[1])])
    CIs = CIs[(density >= densityRange[0]) & (density <= densityRange[1])]
    trueEtas = np.asarray(trueEtas[(density >= densityRange[0]) & (density <= densityRange[1])])
    tumourProp = tumourProp[(density >= densityRange[0]) & (density <= densityRange[1])]
    density = density[(density >= densityRange[0]) & (density <= densityRange[1])]
    
    predictedEtas = np.asarray(predictedEtas[(tumourProp >= tumourPropRange[0]) & (tumourProp <= tumourPropRange[1])])
    CIs = CIs[(tumourProp >= tumourPropRange[0]) & (tumourProp <= tumourPropRange[1])]
    trueEtas = np.asarray(trueEtas[(tumourProp >= tumourPropRange[0]) & (tumourProp <= tumourPropRange[1])])
    density = density[(tumourProp >= tumourPropRange[0]) & (tumourProp <= tumourPropRange[1])]
    tumourProp = tumourProp[(tumourProp >= tumourPropRange[0]) & (tumourProp <= tumourPropRange[1])]

    
    Ybar = np.mean(predictedEtas)
    SStot = np.sum( (predictedEtas-Ybar)**2 )
    SSres = np.sum( (predictedEtas-trueEtas)**2 )
    r2 = 1 - (SSres/SStot)
    print('R squared = ' + str(r2))
    
    RMSE = np.sqrt(np.sum((trueEtas-predictedEtas)**2)/len(trueEtas))  
    
    plt.figure(figsize=(9,9))
    CIwidth = CIs[:,1]-CIs[:,0]
    plt.scatter(trueEtas,predictedEtas,c=CIwidth,cmap=cm.plasma,vmin=0,vmax=MASTER_RHO_THRESHOLD)#,vmin=min(density),vmax=600)
    plt.xlim([0,MASTER_RHO_THRESHOLD])
    plt.ylim([0,MASTER_RHO_THRESHOLD])
    plt.xlabel(noiseParam)
    plt.ylabel('$\\eta_{'+eta_subscript +'}$')
    plt.show()
    
    plt.figure(figsize=(9,9))
    CIwidth = CIs[:,1]-CIs[:,0]
    CIwidths.append(CIwidth)
    inCI = (predictedEtas >= CIs[:,0]) & (predictedEtas <= CIs[:,1])
    plt.scatter(trueEtas,predictedEtas,c=density,cmap=cm.RdBu,vmin=0,vmax=300)#,vmin=min(density),vmax=600)
    plt.xlim([0,MASTER_RHO_THRESHOLD])
    plt.ylim([0,MASTER_RHO_THRESHOLD])
    plt.colorbar(label='Density')
    plt.xlabel(noiseParam)
    plt.ylabel('$\\eta_{'+eta_subscript +'}$')
    plt.show()
    