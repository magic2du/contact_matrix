import os, sys
from dealFile import *
ToDoList=sys.argv[1]
listOfAll=readListFile(ToDoList)
listOfSuccess=[]
for folders in listOfAll:
	if os.path.exists('/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/'+folders):
		sh='cp -ru /home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/'+folders+' /home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/'
		os.system(sh)
		listOfSuccess.append(folders)
writeListFile('listOfSuccessCopied_'+ToDoList,listOfSuccess)
