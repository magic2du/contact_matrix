import _mysql
from dealFile import *
ddis=readDDIsFile('listOfDDIsHaveOver2InterfacesHave10-20Examples.txt')
newddis=[]
for ddi in ddis:
	[domain1,domain2]=ddi
	if domain1<>domain2:
		newddis.append(domain1+'_int_'+domain2)
writeListFile('listOfDDIsHaveOver2InterfacesHave10-20ExamplesWithDifferentDomain.txt',newddis)
