import sys,numpy
from dealFile import *
#get input file name
# get input list file to a list ddis.

def processDDI(ddi):
	if getListForFamily(ddi) and getListForTopology(ddi):
		print '		The working for ddi is %s' % ddi
		listForFamily=getListForFamily(ddi)
		listForTopology=getListForTopology(ddi)
		return listForFamily,listForTopology
	else:
#		print ddi
		return False
def getListForFamily(ddi):
	#return the list of [auc,p_value]
	#ddi='B12-binding_int_MM_CoA_mutase'
	Folder='/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/'
	summaryFileName=Folder+ddi+'/recallVector.txt'
	try:
		data_file = open(summaryFileName)
		data = []
		aucList=[]
		for line in data_file.readlines():
			line = line.strip()
			aucList=line.split('   ')
			aucList=[float(x) for x in aucList]
			data.extend(aucList)
		data_file.close()
		average = float(sum(aucList))/len(aucList)
		print 'The average AUC for ddi %s at family level is : %f' % (ddi, average)  #print "%-12s %d" % (player, score)
		return data
	except:
		print 'This ddi '+ddi+' does not have FisherM0.summary'
		return False
def getListForTopology(ddi):
	#return the list of [auc,p_value]
	#ddi='B12-binding_int_MM_CoA_mutase'
	Folder='/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/'
	topologyList=readListFile(Folder+ddi+'/topologyList.txt')
	data=[]
	for topology in topologyList:
		summaryFileName=Folder+ddi+'/'+topology+'/recallVector.txt'
		#print summaryFileName
		try:	
			data_file = open(summaryFileName)
			aucList=[]
			for line in data_file.readlines():
				line = line.strip()
				aucList=line.split('   ')
				aucList=[float(x) for x in aucList]
				data.extend(aucList)
			data_file.close()
			average = float(sum(aucList))/len(aucList)
			print 'The average AUC for %s for %s is : %f' % (ddi, topology, average)  #print "%-12s %d" % (player, score)
		except:
			print 'This ddi '+ddi+topology+'does not have recallVector.txt'
			return False
	return data
#start
todoList=sys.argv[1]
ddis=readListFile(todoList)
#initiate variables
sumOfAUCOfFamily=0
sumOfAUCOfTopology=0
listTotalFamily=[]
listTotalTopology=[]
numberOfSuccessDDI=0
Error=''
for ddi in ddis:
	#process ddi
	if processDDI(ddi):
		[listForFamily,listForTopology]=processDDI(ddi)
		listTotalFamily.extend(listForFamily)
		listTotalTopology.extend(listForTopology)
		numberOfSuccessDDI=numberOfSuccessDDI+1
#for item in listTotalFamily:#print average
#	sumOfAUCOfFamily+=item
#average=float(sumOfAUCOfFamily)/len(listTotalFamily)
print 'The average Recall of the %d ddis at family level is %f' % (numberOfSuccessDDI, numpy.average(listTotalFamily))
print 'The standard deviation  of Recall at family level is %f' % numpy.std(listTotalFamily)
print 'The total number of examples at family level is %d' % len(listTotalFamily)
#print listTotalFamily
#for item in listTotalTopology:
#	sumOfAUCOfTopology+=item
#average=float(sumOfAUCOfTopology)/len(listTotalTopology)
#print listTotalTopology
print 'The average Recall of the %d ddis at topology level is %f' % (numberOfSuccessDDI, numpy.average(listTotalTopology))
print 'The standard deviation  of Recall at topology level is %f' % numpy.std(listTotalTopology)
print 'The total number of examples at topology level is %d' % len(listTotalTopology)
#writeListFile('listOfDDIsHave2InterfacesOver15WithDomainDifferent',newDDI)
print 'The  numberOfSuccessDDI: %d' % numberOfSuccessDDI
print Error
