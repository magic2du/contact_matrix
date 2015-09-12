#!/usr/bin/env python
import sys
import IO_class
import scipy.io as sio
import numpy as np
#dump the normalized aaIndex in to a mat. struct as aaIndex

# adding cogent pack path.
cogentPATH=''
sys.path.append(cogentPATH)
from cogent.parse.aaindex import *

#aa index file path 
aaindexFilePath='/home/michael/Dropbox/Project/Contact_maps/AAindex/aaindex1'

#create a parser object
aa1p = AAIndex1Parser()

#create a parser object generater. it returns record objects.
#aaIndex1Objects = aa1p(open(aaindexFilePath))

# features to be extracted from AA index
features =	['ANDN920101', 'ARGP820101','BEGF750101','BUNA790103', 'BHAR880101','BURA740102', 'GEOR030101', 'CHOP780204', 'CHOP780215', 'JOND920102', 'KHAG800101', 'FAUJ880104', 'PALJ810107',
'RACS820114','WERD780103', 'YUTK870102', 'CHAM830102']

# 20 amino acid code
f=IO_class.FileOperator('aaList.txt')
aaList=f.readStripLines()
aaList.sort()


def dumpFeatureVectureMAT(): # dump the normalized aaIndex in to a mat. struct as aaIndex
    aaIndex={}
    aaMatrix=[]
    for aa in aaList:
        a, b=get_feature_vector(aa)
        aaFeatureArray=np.array(a)
        print aa, aaFeatureArray
        aaIndex[aa]=aaFeatureArray
        aaMatrix.append(list(aaFeatureArray))
    aaMatrix=np.mat(aaMatrix) # matrix of 20*17
    mean, std, newaaMatrix=normorlizeList(aaMatrix)
    i=0
    for aa in aaList:
        aaIndex[aa]=np.array(newaaMatrix[i,:])
        i+=1
#    print aaIndex['C']
    sio.savemat('aaIndex.mat', {'aaIndex': aaIndex})
def normorlizeList(aaMatrix):
    newaaMatrix=np.zeros(aaMatrix.shape)
    m,n=aaMatrix.shape
    mean=aaMatrix.mean(axis=0)
    std=aaMatrix.std(axis=0)
    for j in range(m):
        newaaMatrix[j, :]=(aaMatrix[j, :]-mean[0])/std[0]
    return   np.array(mean), np.array(std), newaaMatrix  
def get_feature_vector(aa):
    aa=aa.upper()
    feature_vector=[]
    feature_names=[]
    #create a parser object generater. it returns record objects.
    aaIndex1Objects = aa1p(open(aaindexFilePath))
    if not aa.isalpha():
	    raise(aa+' is not a char')
    for record in aaIndex1Objects:
        if record.ID in features:
            feature_vector.append(record.Data[aa])
            feature_names.append(record.ID)
    return feature_vector, feature_names
if __name__ == '__main__':
	#testing case
    feature_vector, feature_names=get_feature_vector('K')

    print feature_vector, feature_names, len(feature_names)
    dumpFeatureVectureMAT()
