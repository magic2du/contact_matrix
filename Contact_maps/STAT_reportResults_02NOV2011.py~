import sys,os
from IO_class import FileOperator
import numpy as np
#get input file name
reportFile=sys.argv[1]
reportFileObj=FileOperator(reportFile)
lines=reportFileObj.readStripLines()

#initiate variables
listOfddi=[]
listOfAUC=[]
listOfsvmRecall=[]
listOfsvmPrecision=[]
listOfbaselineAUC=[]
listOfbaselineRecall=[]
listOfbaselinePrecision=[]

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
