'''
Created on Apr 10, 2013

@author: du
'''
import os,re
from collections import OrderedDict
class FileOperator:
    def __init__(self, fileName):       #file name to be operated.
        self.fileName=fileName

    def read(self):
        if not os.path.exists(self.fileName):
            raise                     #return file content as text
        f=open(self.fileName, 'r')
        data=f.read()
        f.close()
        return data
    def readLines(self):
        f=open(self.fileName, 'r')      #return file content as a list of lines
        data=f.readlines()
        f.close()
        print "number of lines in %s:" %self.fileName +str(len(data)) 
        return data
    def readStripLines(self):
        f=open(self.fileName, 'r')      #return file content as a list of striped lines
        data=[]
        for line in f.readlines():
            line=line.strip()
            data.append(line)
        f.close()
        print "number of lines in %s:" %self.fileName +str(len(data)) 
        return data
    def grepWord(self, string):
        lines=self.readStripLines()
        grepedLines=[line for line in lines if re.findall(string, line)]
        return grepedLines
    def writeList(self,lst):    # write lst into filename.
        data_file = open(self.fileName,'w')
        for item in lst:
            data_file.write("%s\n" % item)
        print "number of lines wrote in %s:" %self.fileName +str(len(lst))
        data_file.close()	
    
class FilesInFolder:                    #return the list of files with path recursively 
    def __init__(self, FOLDER):         #read filename
        self.FOLDER=FOLDER
    def getFiles(self):                 #return the list of files with path recursively 
        data=[]
        for dirname, dirnames, filenames in os.walk(self.FOLDER):
            # print path to all subdirectories first.
           
    #        for subdirname in dirnames:
    #            print os.path.join(dirname, subdirname)
        
            # print path to all filenames.
            for filename in filenames:
                data.append(os.path.join(dirname, filename))
        return data


class DictToCSVWriter:  # tab delimiter to filename of dictionary CSV file
    def __init__(self, filename, dictionary):         #read filename
        self.filename=filename
        self.dictionary=dictionary  
    def write(self):         #read filename
        with open(self.filename, 'w') as f:  # This creates the file object for the context
                                                        # below it and closes the file automatically
            for k, v in self.dictionary.iteritems(): # Iterate over items returning key, value tuples
                f.write('%s\t%s\n' % (str(k), str(v))) # write in to file.


class CSVToDictReader:  # tab delimiter to filename of dictionary CSV file
    def __init__(self, filename):         #read filename
        self.filename=filename 
    def read(self):         #read filename
        dictionary={}
        with open(self.filename, 'r') as f:  # This creates the file object for the context
            for line in f.readlines():
                k,v=line.split('\t')
                dictionary[k]=v.strip()
        return dictionary
class DLLogFileOperator(FileOperator):
    def __init__(self, filename):         #read filename
        self.fileName=filename
        self.resultTable=self.getResultTable()
        self.DDIs=self.getDDIs()
        self.aucMean=self.getAUCMean()
        self.accendingDDIsAccodingAUC= self.sortDDIaccordingToTheirAUC()  
    def getScoreList(self, scoreString):
        logContent=self.read()
        regPattern=scoreString+' = (\d+.\d+)'
        auc = re.findall(regPattern, logContent)
        aucList=[float(item) for item in auc]
        return aucList
    def plotAUC(self):
        pass
    def grepLinesWithResult(self):
        resultLines=self.grepWord('seqPair')
        return resultLines
    def getDDIs(self):
        resultLines=self.grepLinesWithResult()
        listOfDDI=set()
        for line in resultLines:
            splitted=line.split(', ')
            listOfDDI.add(splitted[0])
        listOfDDI=list(listOfDDI)
        listOfDDI.sort()
        return listOfDDI
    def getResultTable(self):
        resultTable=[]
        resultLines = self.grepLinesWithResult()
        for line in resultLines:
            splitted=line.split(', ')
            DDI=splitted[0]
            sequence_num=int(splitted[1].split("seqPair ")[1])
            auc=float(splitted[2].split("auc = ")[1])
            sens=float(splitted[3].split("sens = ")[1])
            spec=float(splitted[4].split("spec = ")[1])
            blAUC=float(splitted[5].split("blAUC = ")[1])
            blSens=float(splitted[6].split("blSens = ")[1])
            blSpec=float(splitted[7].split("blSpec = ")[1][:-1])
            d=OrderedDict()
            d['DDI']=DDI
            d['sequence_num']=sequence_num
            d['auc']=auc
            d['sens']=sens
            d['spec']=spec
            d['blAUC']=blAUC
            d['blSens']=blSens
            d['blSpec']=blSpec
            resultTable.append(d)
        #print resultTable
        resultTable.sort(key=lambda row: row["DDI"])
        print resultTable
        return resultTable                
    def getScoresForGivenDDI(self, DDI, scoreString):
        return [row[scoreString] for row in self.resultTable if row["DDI"]==DDI]
    def getScoresMeanForGivenDDI(self, DDI, scoreString):
        scoreList=self.getScoresForGivenDDI(DDI, scoreString)
        return sum(scoreList)/len(scoreList)
    def getAUCMean(self):
        aucList=[]
        for DDI in self.DDIs:
            aucList.extend(self.getScoresForGivenDDI(DDI, 'auc'))
        return sum(aucList)/len(aucList)
    def sortDDIaccordingToTheirAUC(self):
        sortedDDIs=sorted(self.DDIs, key=lambda ddi: self.getScoresMeanForGivenDDI(ddi, 'auc'))
        return sortedDDIs
    def sortDDIaccordingToTheirAUCminusblAUC(self):
        sortedDDIs=sorted(self.DDIs, key=lambda ddi: self.getScoresMeanForGivenDDI(ddi, 'auc')-self.getScoresMeanForGivenDDI(ddi, 'blAUC'))
        return sortedDDIs
    def dumpTableToCVS(self, table):
        csvFileName = self.fileName+'.csv'
        listOfContent=[]
        firstRow=table[0]
        firstline=','.join([k for k in firstRow])
        listOfContent.append(firstline)
        for row in table:
            contentList=[str(row[key]) for key in row]
            line=','.join(contentList)
            listOfContent.append(line)
        for DDI in self.accendingDDIsAccodingAUC:
            print DDI, self.getScoresMeanForGivenDDI(DDI, "auc")
            line=','.join([DDI, str(self.getScoresMeanForGivenDDI(DDI, "auc"))])
            listOfContent.append(line)
        fileObj=FileOperator(csvFileName)
        fileObj.writeList(listOfContent)
class CompareTwoTable():
    def __init__(self, logObjA, logObjB):         #read filename
        self.logObjA=logObjA
        self.logObjB=logObjB
        self.tableA=logObjA.resultTable
        self.tableB=logObjB.resultTable
        self.elementWiseTable=self.elementWiseComparision()
        self.accendingDDIsAccodingdiffAUC= self.sortDDIaccordingToTheirAUCminusblAUC()
    def elementWiseComparision(self):
        newTable=[]
        for rowA in self.tableA:
            d=OrderedDict()
            for rowB in self.tableB:
                if (rowA['DDI']==rowB['DDI'] and rowA['sequence_num']==rowB['sequence_num']):
                    d['DDI']=rowA['DDI']
                    d['sequence_num']=rowA['sequence_num']
                    d['auc']=rowA['auc']
                    d['blAUC']=rowB['auc']
                    d['diffAUC']=rowA['auc']-rowB['auc']
                    newTable.append(d)
        return newTable
    def getScoresForGivenDDI(self, DDI, scoreString):
        return [row[scoreString] for row in self.elementWiseTable if row["DDI"]==DDI]
    def getScoresMeanForGivenDDI(self, DDI, scoreString):
        scoreList=self.getScoresForGivenDDI(DDI, scoreString)
        return sum(scoreList)/len(scoreList)
    def sortDDIaccordingToTheirAUCminusblAUC(self):
        sortedDDIs=sorted(self.logObjA.DDIs, key=lambda ddi: self.getScoresMeanForGivenDDI(ddi, 'diffAUC'))
        return sortedDDIs
    def dumpTableToCVS(self, fileName):
        csvFileName = fileName+'.csv'
        listOfContent=[]
        table=self.elementWiseTable
        firstRow=table[0]
        firstline=','.join([k for k in firstRow])
        listOfContent.append(firstline)
        for row in table:
            contentList=[str(row[key]) for key in row]
            line=','.join(contentList)
            listOfContent.append(line)
        for DDI in self.accendingDDIsAccodingdiffAUC:
            print DDI, self.getScoresMeanForGivenDDI(DDI, "diffAUC")
            line=','.join([DDI,str(self.getScoresMeanForGivenDDI(DDI, "auc")), str(self.getScoresMeanForGivenDDI(DDI, "blAUC")), str(self.getScoresMeanForGivenDDI(DDI, "diffAUC"))])
            listOfContent.append(line)
        fileObj=FileOperator(csvFileName)
        fileObj.writeList(listOfContent)
