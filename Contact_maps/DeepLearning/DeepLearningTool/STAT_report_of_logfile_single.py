import sys,os
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
file1="log_DL_RE_US_AA11_1_8_200.txt"
#file1="log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_DL_RE_US_DL_RE_US_1_1_29JAN2014.txt"
reportFileObj=DLLogFileOperator(file1)
listOfAUC1=reportFileObj.getScoreList('auc')
#reportFileObj.getResultTable()

print reportFileObj.DDIs
#print reportFileObj.accendingDDIs
for DDI in reportFileObj.sortDDIaccordingToTheirAUCminusblAUC():
    scoreDiff=reportFileObj.getScoresMeanForGivenDDI(DDI, "auc")-reportFileObj.getScoresMeanForGivenDDI(DDI, "blAUC")

    print DDI, scoreDiff
    print reportFileObj.getScoresMeanForGivenDDI(DDI, "auc"), reportFileObj.getScoresMeanForGivenDDI(DDI, "blAUC")
reportFileObj.dumpTableToCVS(reportFileObj.resultTable)
#reportFileObj=DLLogFileOperator(file2)
#listOfAUC2=reportFileObj.getScoreList('auc')



#initiate variables
listOfddi=[]
listOfAUC=[]
listOfsvmRecall=[]
listOfsvmPrecision=[]
listOfbaselineAUC=[]
listOfbaselineRecall=[]
listOfbaselinePrecision=[]
'''
for line in lines:
    [ddi, seqNumber, AUC, svmRecall, svmPrecision,
    baselineAUC, baselineRecall, baselinePrecision]= line.split()
    listOfddi.append(ddi)
    listOfAUC.append(float(AUC))
    listOfsvmRecall.append(float(svmRecall))
    listOfsvmPrecision.append(float(svmPrecision))
    listOfbaselineAUC.append(float(baselineAUC))
    listOfbaselineRecall.append(float(baselineRecall))
    listOfbaselinePrecision.append(float(baselinePrecision))
    
print 'The total number of DDIs is  %d ' % len(set(listOfddi))
print 'The number of training sequence pairs LOO is  %d ' % len(lines)
print 'The number of average SVM AUC is  %f ' % np.mean(listOfAUC)
print 'The number of average SVM Recall is  %f ' % np.mean(listOfsvmRecall)
print 'The number of average SVM Precision is  %f ' % np.mean(listOfsvmPrecision)
print 'The number of average baseline AUC is  %f ' % np.mean(listOfbaselineAUC)
print 'The number of average baseline Recall is  %f ' % np.mean(listOfbaselineRecall)
print 'The number of average baseline Precision is  %f ' % np.mean(listOfbaselinePrecision)
'''
