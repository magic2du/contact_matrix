import sys,os
sys.path.append("/home/du/Dropbox/Project/Contact_maps/")
#from scipy.stats import ttest_1samp, ttest_ind
from IO_class import *
#get input file name
'''
file1='/home/michael/Dropbox/Project/Contact_maps/DeepLearning/DeepLearningTool/log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1_SVMLIGHT_LINEAR_1_1_13NOV2013.txt'
file2='/home/michael/Dropbox/Project/Contact_maps/DeepLearning/DeepLearningTool/log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_SVMLIGHT_LINEAR_1_1_13NOV2013.txt'
'''
#file1='log_FisherM1ONLY_SVMLIGHT_POLY_30_89.222.txt'
#file2='log_FisherM1ONLY_SVMLIGHT_LINEAR_30_90.702.txt'
#file1='log_FisherM1ONLY_SVMLIGHT_LINEAR_30_90.702.txt'
#file2='log_FisherM1ONLY_DLSTOP_30_90.18.txt'
#file2='log_FisherM1ONLY_DL_30_90.569.txt'
#file2='log_FisherM1ONLY_SAESVM_L2_90.411.txt'
#file1='log_FisherM1AA11_SVMLIGHT_84.002.txt'
#file1='log_FisherM1AA11_DL_30_86.13.txt'

#file2='log_FisherM1ONLY_DL_CLUSTER_US_SAE_US_SVM_90.209.txt'
#file2='log_FisherM1ONLY_DLSTOP_L2_84.344.txt'
#


#file1='log_Single_FisherM1ONLY_SVMLIGHT_96.12.txt'
#file2='log_Single_FisherM1_SVMLIGHT_98.537.txt'

#file1='log_MultiTop_FisherM1ONLY_SVMLIGHT_95.525.txt'
file1='log_MultiTop_FisherM1AA11_SVMLIGHT_97.689.txt'
#file2='log_MultiTop_FisherM1ONLY_DLSTOP_96.419.txt'
file2='log_MultiTop_FisherM1_DLSTOP_98.181.txt'
#file2='log_MultiTop_FisherM1AA11_DL_97.885.txt'
#file2='log_MultiTop_FisherM1ONLY_DL_96.392.tt'
#file1='log_FisherM1ONLY_DLUS_Unlabed_1_8th_87.599.txt'
#file1='log_FisherM1ONLY_DLUS_Unlabeled_1_8th_86.844.txt'
#file2='log_FisherM1ONLY_DL_reduced_1_8th_85.535.txt'
#file2='log_FisherM1AA1_SVMLIGHT_POLY_30_82.56.txt'

if len(sys.argv)>1:
    pass
reportFileObj=DLLogFileOperator(file1)
listOfAUC1=reportFileObj.getScoreList('auc')
#reportFileObj.getResultTable()
reportFileObj.accendingDDIsAccodingAUC

print reportFileObj.accendingDDIsAccodingAUC
for DDI in reportFileObj.accendingDDIsAccodingAUC:
    print DDI, reportFileObj.getScoresMeanForGivenDDI(DDI, "auc")
reportFileObj.dumpTableToCVS(reportFileObj.resultTable)
#######file 2##########
reportFileObj2=DLLogFileOperator(file2)
listOfAUC1=reportFileObj2.getScoreList('auc')
#reportFileObj.getResultTable()
reportFileObj2.accendingDDIsAccodingAUC

print reportFileObj2.accendingDDIsAccodingAUC
for DDI in reportFileObj2.accendingDDIsAccodingAUC:
    print DDI, reportFileObj2.getScoresMeanForGivenDDI(DDI, "auc")
reportFileObj2.dumpTableToCVS(reportFileObj2.resultTable)
compareOjb=CompareTwoTable(reportFileObj, reportFileObj2)
compareOjb.dumpTableToCVS("Comp_"+file1+file2)
'''
for DDI in reportFileObj.sortDDIaccordingToTheirAUCminusblAUC():
    scoreDiff=reportFileObj.getScoresMeanForGivenDDI(DDI, "auc")-reportFileObj.getScoresMeanForGivenDDI(DDI, "blAUC")
    print reportFileObj.getScoresMeanForGivenDDI(DDI, "auc"), reportFileObj.getScoresMeanForGivenDDI(DDI, "blAUC")
    print DDI, scoreDiff
reportFileObj.dumpTableToCVS(reportFileObj.resultTable)
#reportFileObj=DLLogFileOperator(file2)
#listOfAUC2=reportFileObj.getScoreList('auc')
'''


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
