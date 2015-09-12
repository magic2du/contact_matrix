from dealFile import *
import sys
logFile=sys.argv[1]

#listOfAll=readListFile('ddisToRunRetrListOfErrorsAndBigMay9.txt')
grepLogFile(logFile)
listOfAllSuccess=readGrepFile('SUCCESS_'+logFile)
listOfAllError=readGrepFile('ERROR_'+logFile)
print(len(listOfAllSuccess))
#writeListFile(filename,lst)
writeListFile('list_of_SUCCESS_'+logFile,listOfAllSuccess)
