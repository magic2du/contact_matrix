import _mysql
from dealFile import *
ddis=readDDIsFile('listOfDDIsAtLeast1InterfacesHave15-20Examples.txt')
newddis=[]
for ddi in ddis:
	[domain1,domain2]=ddi
	if domain1<>domain2:
		newddis.append(domain1+'_int_'+domain2)
writeListFile('listOfDDIsAtLeast1InterfacesHave15-20ExamplesWithDifferentDomain.txt',newddis)
