import sys,numpy
from dealFile import *
#get input file name
# get input list file to a list ddis.

def getListForTopology(ddi):
	#return the list of [auc,p_value]
	#ddi='B12-binding_int_MM_CoA_mutase'
	Folder='/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/'
	topologyList=readListFile(Folder+ddi+'/topologyList.txt')
	return topologyList
#start
todoList=sys.argv[1]
ddis=readListFile(todoList)
#initiate variables
sumOfAUCOfFamily=0
sumOfAUCOfTopology=0
listTotalFamily=[]
listTotalTopology=[]
numberOfSuccessDDI=0
for ddi in ddis:
	listTotalTopology.extend(getListForTopology(ddi))
#for item in listTotalFamily:#print average
#	sumOfAUCOfFamily+=item
#average=float(sumOfAUCOfFamily)/len(listTotalFamily)
print 'The number of DDIs is %d' % len(ddis)

print 'The total number of topologies in the %d DDIs is %d' % (len(ddis), len(listTotalTopology))