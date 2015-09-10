import _mysql
from dealFile import *
#Get of Domains which has more than 2 interfaces have 16-20 examples
db=_mysql.connect(host="localhost",user="root",passwd="zxcv4321",db="DDI")
#db.query("""select COUNT(*) from PPI inner join example on (ID = PPI_ID) where domain1="ACT" and domain2="ACT" and topology_1 = 6 and topology_2 = 6""")
#db.query("""select * from PPI inner join example on (ID = PPI_ID) where domain1="ACT" and domain2="ACT" """)
ddiList=readDDIsFile('listOfDDIsHave2InterfacesOver15.txt')
ddis=[]
#Number of Domains which has 2 interfaces have more than 15 examples
for ddi in ddiList:
    [domain1,domain2]=ddi
    #print i
    print domain1
    print domain2
    #query='SELECT DISTINCT topology_1,topology_2 from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
    #query='SELECT DISTINCT topology_1,topology_2 from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
    query='SELECT COUNT(DISTINCT topology_1,topology_2) from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
    #print query
    #query='select domain1,domain2 from DDI1'
    db.query(query)
    result=db.store_result()
    numTopology=result.fetch_row(0)
    #print numTopology[0][0]
    if int(numTopology[0][0])>1:
	Ctr=False
	query='SELECT DISTINCT topology_1,topology_2 from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
	db.query(query)
	result=db.store_result()
	rTopology=result.fetch_row(0)
	numOver15=0
	for val in rTopology[0:]:
            [topology1,topology2]=val
            try:
                #print topology1+':'+topology2
                query='SELECT COUNT(*) from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'" AND topology_1='+topology1+' AND topology_2='+topology2
                print query
                db.query(query)
                result=db.store_result()
                numExample=result.fetch_row(0)
                print numExample[0][0]
            except:
                break            
            if int(numExample[0][0])>15:# if for those interfaces have more than 15 examples if they have less than 21 examples, add it.
		if int(numExample[0][0])>20:
			Ctr=False
		        break
		else:
			Ctr=True
	if Ctr==True:
		ddis.append(domain1+'_int_'+domain2)

writeListFile('listOfDDIsHaveOver2InterfacesHave15-20Examples.txt',ddis)
#print result.fetch_row()
#print r[0][0] readDDIsFile('listOfDDIsHave2InterfacesOver15.txt')
