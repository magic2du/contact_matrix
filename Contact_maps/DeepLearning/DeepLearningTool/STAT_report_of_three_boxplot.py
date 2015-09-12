import sys,os
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes
from scipy import stats
sys.path.append("/home/michael/Dropbox/Project/Contact_maps/")
#from scipy.stats import ttest_1samp, ttest_ind
from IO_class import DLLogFileOperator
#get input file name
'''
file1='/home/michael/Dropbox/Project/Contact_maps/DeepLearning/DeepLearningTool/log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1_SVMLIGHT_LINEAR_1_1_13NOV2013.txt'
file2='/home/michael/Dropbox/Project/Contact_maps/DeepLearning/DeepLearningTool/log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_SVMLIGHT_LINEAR_1_1_13NOV2013.txt'
'''
#file1='log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1_DL_RE_US_DL_1_1_17DEC2013.txt'
#file2='log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_SVMLIGHT_LINEAR_1_1_14NOV2013.txt'
#file1='log_MultiTop_FisherM1ONLY_SVMLIGHT_95.525.txt'
#file1="log_DL_RE_US_AA11_1_8_200.txt"
file1="log_DLSTOP_225_99.146.txt"
file2=""
#file1="log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_DL_RE_US_DL_RE_US_1_1_29JAN2014.txt"
reportFileObj=DLLogFileOperator(file1)
listOfAUC1=reportFileObj.getScoreList('auc')
#reportFileObj.getResultTable()

print reportFileObj.DDIs
#print reportFileObj.accendingDDIs
'''
for DDI in reportFileObj.sortDDIaccordingToTheirAUCminusblAUC():
    scoreDiff=reportFileObj.getScoresMeanForGivenDDI(DDI, "auc")-reportFileObj.getScoresMeanForGivenDDI(DDI, "blAUC")

    print DDI, scoreDiff
    print reportFileObj.getScoresMeanForGivenDDI(DDI, "auc"), reportFileObj.getScoresMeanForGivenDDI(DDI, "blAUC")
'''
#reportFileObj.dumpTableToCVS(reportFileObj.resultTable)
table1=reportFileObj.getResultTable()
reportFileObj=DLLogFileOperator(file2)
table2=reportFileObj.getResultTable()
auc1=[]
auc2=[]
for item1 in table1:
	auc1.append(item1['auc'])

np.mean(auc1)
np.mean(auc2)
stats.ttest_rel(auc1,auc2)# paired t test.

diff=np.array(auc1)-np.array(auc2)
my_std = np.std(diff) 
my_mean = np.mean(diff)
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')
# Some fake data to plot
A= [auc1,  auc2]
fig = figure()
ax = axes()
hold(True)
bp = boxplot(A, positions = [1, 2], widths = 0.6)
#setBoxColors(bp)
# set axes limits and labels
ylim(0.8,1.05)
ax.set_xticklabels(['Deep Learning', 'SVM'])
title('Comparision between Deep Learning and SVM')
ylabel('AUC Score')
savefig('boxcompare.png')
show()

