import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.5)
from utilities import *
import scipy.stats as ss
from scipy.optimize import curve_fit
sns.set_style('white')


def func_negExponential(x, a, b,c):
    # Helper function for plotting a negative exponential curve
    return a*np.exp(-b*x) + c

#%% Generate lookup table
# Restrict density range of data points
densityRange = [150,10000]
# Restrict data to GRF with proportion of tumour area within this range (vs stroma)
tumourPropRange = [0.25,0.75]
# Restrict range of rho considered
MASTER_RHO_THRESHOLD = 0.5


eta = np.arange(0,MASTER_RHO_THRESHOLD+0.02,0.02)

trainingDataFile = 'DataFiles/SyntheticData_TrainingDataset.csv'
df_train = pd.read_csv(trainingDataFile)
df_train = df_train[df_train['Rho'] <= MASTER_RHO_THRESHOLD]
df_train = df_train[(df_train['True Density per mm squared'] >= densityRange[0]) & (df_train['True Density per mm squared'] <= densityRange[1])]
df_train = df_train[(df_train['Tumour Area (mm2)']/2.25 >= tumourPropRange[0]) & (df_train['Tumour Area (mm2)']/2.25 <= tumourPropRange[1])]


# Coefficients from Figure 3 (b-d)

meanCoefficients = [[1.337835446744767, 4.524226183972446, 1.102179018392261],
                    [0.18220190564950053, 9.16345960231854, 0.0919879950991511],
                    [-2.0488935134209423, 0.7428606945640334, 2.055300770901428]]

SDCoefficients = [[0.7822985714408306, 8.263005530901681, 0.05881049711296449],
                   [0.12625320564813167, 39.544681027423124, 0.026580098150134346],
                   [-0.5690804644179286, 1.462391671332397, 0.5665079098909845]]

# Use helper function to establish curves used in Fig 3 (b-d)
Jmin_means = func_negExponential(eta,meanCoefficients[0][0],meanCoefficients[0][1],meanCoefficients[0][2])
Fmax_means = func_negExponential(eta,meanCoefficients[1][0],meanCoefficients[1][1],meanCoefficients[1][2])
gmax_means = func_negExponential(eta,meanCoefficients[2][0],meanCoefficients[2][1],meanCoefficients[2][2])

Jmin_SDs = func_negExponential(eta,SDCoefficients[0][0],SDCoefficients[0][1],SDCoefficients[0][2])
Fmax_SDs = func_negExponential(eta,SDCoefficients[1][0],SDCoefficients[1][1],SDCoefficients[1][2])
gmax_SDs = func_negExponential(eta,SDCoefficients[2][0],SDCoefficients[2][1],SDCoefficients[2][2])

Jrange = np.arange(0,1,0.01)
Frange = np.arange(0,1.5,0.01)
grange = np.arange(1,5,0.01)
means = [gmax_means,Fmax_means,Jmin_means]
SDs = [gmax_SDs,Fmax_SDs,Jmin_SDs]
ranges = [grange,Frange,Jrange]

#%% Now read in labelled data and make boxplots

# Use p-values or symbols in plot
pValuesAsSymbol = False
             
df_manual = pd.read_csv('DataFiles/InfiltrationManualScores.csv')


colors = plt.cm.plasma(np.linspace(0,1,4))
pal = sns.color_palette("plasma",4)
data = []
for i in range(len(df_manual)):
    score = df_manual.Score[i]
    
    dataLine = []
    if score == 0:
        dataLine.append('Very Low')
    elif score == 1:
        dataLine.append('Low')
    elif score == 2:
        dataLine.append('Moderate')
    elif score == 3:
        dataLine.append('High')
    else:
        assert(1==2) 
    
    dataLine.extend([score,df_manual.maxPCF[i],df_manual.maxSCDr[i],df_manual.minJ[i],df_manual.Density[i]])
    data.append(dataLine)
    
# Put data into dataframe
df_boxplots = pd.DataFrame(data)
df_boxplots = df_boxplots.rename({0:"Manual Score",1:"ManualScoreNumeric",2:"$g_{max}$",3:"$F_{max}$",4:"$J_{min}$",5:"Density"}, axis='columns')

labels = ['$g_{max}$',"$F_{max}$","$J_{min}$","Density"]
observations = [df_manual.maxPCF,df_manual.maxSCDr,df_manual.minJ]
# Which combinations of labels do we plot?
testSets = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]

trainingRho = [df_train['Rho']]
trainingStats = [df_train['maxPCF'],df_train['maxSCDr'],df_train['minJ']]

# Resolution of g_max, F_max and J_min to generate likelihood over
g_grid = np.arange(1,5,0.01)
F_grid = np.arange(0,1.5,0.01)
J_grid = np.arange(0,1,0.01)
rangesToGenerateTable = [g_grid, F_grid, J_grid]

etasLog = []
for i, value in enumerate(testSets):
    indices = testSets[i]
    
    r = [rangesToGenerateTable[v] for v in indices]
    m = [meanCoefficients[v] for v in indices]
    s = [SDCoefficients[v] for v in indices]
    trainingData = [trainingStats[v] for v in indices]
    trainingData = np.concatenate((trainingRho,trainingData),axis=0)
    
    etaValues, logLikelihoodTable = GenerateLogLikelihoodTable_FittedMeanAndSD(trainingData, r, m, s)

    predictedEtas = []
    CIs = []
    n = len(indices)
    for i in range(len(df_boxplots)):
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
    etasLog.append(predictedEtas)
    sns.set(style='white',font_scale=2)

    print(indices)
    titl = ', '.join([labels[v] for v in indices])
    eta_subscript = ', '.join([labels[v][1] for v in indices])
    
    plt.figure(figsize=(9,9))
    sns.boxplot(x="Manual Score",y=predictedEtas,data=df_boxplots,palette=pal,order=["Very Low","Low","Moderate","High"])
    plt.ylabel('$\eta_{' + eta_subscript + '}$')
    # statistical annotation
    pairs = [[0,1],[1,2],[2,3]]
    for j in range(len(pairs)):
        a = pairs[j][0]
        b = pairs[j][1]
        data1 = predictedEtas[df_boxplots.ManualScoreNumeric == a]
        data2 = predictedEtas[df_boxplots.ManualScoreNumeric == b]
        
        t,p=ss.ttest_ind(data1,data2)
        print(p)
        if p < 0.05:
            x1, x2 = a, b   # column numbers
            y, h, col = np.max([np.max(data1),np.max(data2)]) + 0.025, 0.025, 'k'
            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
            
            if pValuesAsSymbol:
                p_sym = 'n.s.'
                if p < 0.05:
                    p_sym = '$\\ast$'
                if p < 0.01:
                    p_sym = '$\\ast \\ast$'
                if p < 0.001:
                    p_sym = '$\\ast \\ast \\ast$'
                plt.text((x1+x2)*.5, y+h, p_sym, ha='center', va='bottom', color=col,fontsize=12)
            else:
                plt.text((x1+x2)*.5, y+h, "p = " + '%.2g' % p, ha='center', va='bottom', color=col,fontsize=18)
    

    # Same, but vertically and sorted by pathologist classification
    plt.figure(figsize=(9,9))
    integers = range(0,len(predictedEtas))
    order = []
    for score in range(4):
        filt = df_boxplots[df_boxplots.ManualScoreNumeric == score]
        filterEtas = predictedEtas[df_boxplots.ManualScoreNumeric == score]
        indices = np.asarray(filt.index)
        indOrder = np.argsort(filterEtas)
        order.extend(indices[indOrder])
    sortedCIs = np.asarray(CIs)[order]
    sortedScores = df_boxplots.ManualScoreNumeric[order]
    colVec = []
    for i in range(len(CIs)):
        c = pal[sortedScores.iloc[i]]
        colVec.append(c)
        ends = sortedCIs[i]
        plt.plot([integers[i], integers[i]],[ends[0], ends[1]], linewidth=2,c=c)
    plt.scatter(integers,predictedEtas[order],c=colVec)
    plt.ylabel('$\eta_{' + eta_subscript + '}$')
    plt.gca().set_xticks([])
    sns.despine()
    