import os 
def readListFile(filename):#read file:filename lines into list
	data_file = open(filename)
	data = []
	for line in data_file.readlines():
		line = line.strip()
		#print line
		data.append(line)
	data_file.close()
	print "number of lines in %s:" %filename +str(len(data)) 
	return data
def readGrepFile(filename):#read ddis greped log File  into list.ie SUCCESS_log_file
	data_file = open(filename)
	data = []
	for line in data_file.readlines():
		temp1 = line.split(" ")
		line = temp1[3].split(",")
		line=line[0]
		line=line.strip()
		data.append(line)
	'''   print(line)
	   line = line.split(" ")
	   t1 = line[1].split(":")
	   t2 = line[2].split(":")
	   tmp = [float(t1[1]), float(t2[1]), int(line[0])]
	   data.append(tmp)
	   print tmp'''
	print "number of lines in %s:" %filename +str(len(data))
	data_file.close()
	return data
def writeListFile(filename,lst):# write lst into filename.
	data_file = open(filename,'w')
	for item in lst:
		data_file.write("%s\n" % item)
	print "number of lines wrote in %s:" %filename +str(len(lst))
	data_file.close()
def dealLogFile(filename,listfile):
#grep log file(filename) to ERROR and SUCCESS file: Get the Notfinished ddis in todolist file (listfile) write into NotFinished_log
	sh='grep ERROR: '+filename+'>ERROR_'+filename
	sh2='grep SUCCESS: '+filename+'>SUCCESS_'+filename
	os.system(sh)
	os.system(sh2)
	List1=readListFile(listfile)
	List2=readGrepFile('SUCCESS_'+filename)
	List3=readGrepFile('ERROR_'+filename)
	List4=list(set(List1)-set(List2)-set(List3))
	writeListFile('NotFinished_'+filename,List4)
def grepLogFile(filename):
#grep log file(filename) to ERROR and SUCCESS file: 
	sh='grep ERROR: '+filename+'>ERROR_'+filename
	sh2='grep SUCCESS: '+filename+'>SUCCESS_'+filename
	os.system(sh)
	print sh
	os.system(sh2)
	print sh2
def readDDIsFile(filename):#read ddi file into a list with two domain names seperated.
	data_file = open(filename)
	data = []
	for line in data_file.readlines():
		line = line.strip()
		try:
		    [domain1,domain2]=line.split('_int_')
		    data.append([domain1,domain2])
		except:
		    print line
		
	data_file.close()
	print "number of ddis in %s:" %filename +str(len(data)) 
	return data	
