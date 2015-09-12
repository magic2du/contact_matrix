import _mysql
from dealFile import *
#Get of Domains which has more than 2 interfaces have 16-20 examples
db=_mysql.connect(host="localhost",user="root",passwd="zxcv4321",db="DDI")
#db.query("""select COUNT(*) from PPI inner join example on (ID = PPI_ID) where domain1="ACT" and domain2="ACT" and topology_1 = 6 and topology_2 = 6""")
#db.query("""select * from PPI inner join example on (ID = PPI_ID) where domain1="ACT" and domain2="ACT" """)
ddiList=readDDIsFile('listOfDDIsOver2InterfacesOver9.txt')
ddis=[]
#Number of Domains which has 2 interfaces have more than 15 examples
for ddi in ddiList:
    [domain1,domain2]=ddi
    #print i
    #print domain1
    #print domain2
    #query='SELECT DISTINCT topology_1,topology_2 from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
    #query='SELECT DISTINCT topology_1,topology_2 from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
    query='SELECT COUNT(DISTINCT topology_1,topology_2) from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
    #print query
    #query='select domain1,domain2 from DDI1'
    db.query(query)
    result=db.store_result()
    numTopology=result.fetch_row(0)
    print numTopology[0][0]
    if numTopology[0][0]<2:
        break
    try:
        query='SELECT COUNT(*) from DDItopology WHERE domain1="'+domain1+'" AND domain2="'+domain2+'"'
        #print query
        db.query(query)
        result=db.store_result()
        numExample=result.fetch_row(0)
        print int(numExample[0][0])
        if int(numExample[0][0])>50 and int(numExample[0][0])<150 and domain1!=domain2:
            ddis.append(domain1+'_int_'+domain2)
    except:
        print 'error'
        break     
writeListFile('listOfDDIsHaveOver2InterfacesHave50-150_Examples_No_Same_Names_pseudo.txt',ddis)
#print result.fetch_row()
#print r[0][0] readDDIsFile('listOfDDIsHave2InterfacesOver15.txt')
