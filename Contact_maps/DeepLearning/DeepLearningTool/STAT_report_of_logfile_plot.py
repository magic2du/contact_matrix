import sys,os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
sys.path.append("/home/michael/Dropbox/Project/Contact_maps/")
from scipy import stats
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes
#from scipy.stats import ttest_1samp, ttest_ind
from IO_class import DLLogFileOperator
#get input file name
if len(sys.argv)==2:
	file1 = sys.argv[0]
	'''
	file1='/home/michael/Dropbox/Project/Contact_maps/DeepLearning/DeepLearningTool/log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1_SVMLIGHT_LINEAR_1_1_13NOV2013.txt'
	file2='/home/michael/Dropbox/Project/Contact_maps/DeepLearning/DeepLearningTool/log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_SVMLIGHT_LINEAR_1_1_13NOV2013.txt'
	'''
	#file1='log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1_DL_RE_US_DL_1_1_17DEC2013.txt'
	#file2='log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_SVMLIGHT_LINEAR_1_1_14NOV2013.txt'
	#file1='log_MultiTop_FisherM1ONLY_SVMLIGHT_95.525.txt'
	#file1="log_DL_RE_US_AA11_1_8_200.txt"
	#file1="log_DLSTOP_225_99.146.txt"
	#file2="log_DL_RE_US_AA1_1_4_225_97.762.txt"
else:
	#file1 = "log_run3DID_test_load_DL_remoteFisherM1_DL_RE_US_DL_RE_US_1_1_21APR2014.txt"
	#file1 = 'log_CrossValidation_load_DL_remoteFisherM1_DL_RE_US_DL_RE_US_1_1_03MAY2014.txt1.txt'
	#file1='log_CrossValidation_load_DL_remoteFisherM1_DL_RE_US_DL_RE_US_1_1_01MAY2014.txt'
	#file1='log_CrossValidation_load_DL_remoteFisherM1_DL_RE_US_DL_RE_US_1_1_08MAY2014.txt1.txt'
	file1='log_CrossValidation_load_DL_remoteFisherM1_DL_RE_US_DL_RE_US_1_1_08MAY2014.txt1.txt1.txt1.txt'
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')
#file1="log_run3DID_AllVectorsChooseNegRand_aaIndex_test_load_DL_remoteFisherM1ONLY_DL_RE_US_DL_RE_US_1_1_29JAN2014.txt"
reportFileObj=DLLogFileOperator(file1)
#reportFileObj.getResultTable()

print reportFileObj.DDIs
table=reportFileObj.getResultTable()
for ddi in reportFileObj.DDIs:
    
	AUClist=[]
        blAUClist=[]
	recall=[]
	blRecall=[]
	precision=[]
	blPrecision=[]
        for item in table:
            if ddi==item['DDI']:
		AUClist.append(item['auc'])
		blAUClist.append(item['blAUC'])
		recall.append(item['sens'])
		blRecall.append(item['blSens'])
		precision.append(item['spec'])
		blPrecision.append(item['blSpec'])
	print('current DDI:', ddi, len(AUClist))
	print 'AUC:'
	print np.mean(AUClist)
	print np.mean(blAUClist)
        print (stats.ttest_rel(AUClist,blAUClist))
	print 'Recall:'
	print np.mean(recall)
	print np.mean(blRecall)
        print (stats.ttest_rel(recall,blRecall))
	print 'Precision:'
	print np.mean(precision)
	print np.mean(blPrecision)
        print (stats.ttest_rel(precision,blPrecision))
	# Some fake data to plot
	A= [AUClist,  blAUClist]
	fig = figure()
	ax = axes()
	hold(True)
	bp = boxplot(A, positions = [1, 2], widths = 0.6)
	#setBoxColors(bp)
	# set axes limits and labels
	ylim(0.8,1.05)
	ax.set_xticklabels(['1/8 Unsupervised', 'Reduced'])
	plt.title('Comparison between Unsupervised and Reduced DL: '+ddi)
	plt.ylabel('AUC Score')
	plt.savefig('boxcompare.png')
	show()


