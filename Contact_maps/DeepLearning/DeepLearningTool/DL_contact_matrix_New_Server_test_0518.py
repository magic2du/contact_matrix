
# coding: utf-8

# In[5]:

import sys, os

sys.path.append('../../../libs/')
sys.path.append('../libs/')
sys.path.append('../../libs/')
import os.path
import IO_class
from IO_class import FileOperator
from sklearn import cross_validation
import sklearn
import csv
from dateutil import parser
from datetime import timedelta
from sklearn import svm
import numpy as np
import pdb
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import scipy.stats as ss
from sklearn.svm import LinearSVC
import random
# from DL_libs import *
from itertools import izip #new
import math
from sklearn.svm import SVC

import cPickle
import gzip
import numpy
import os
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import time


from dlnn.io_func.model_io import _nnet2file, _cfg2file, _file2nnet, log
from dlnn.utils.learn_rates import _lrate2file, _file2lrate
from dlnn.models.dnn import DNN
from dlnn.models.dropout_nnet import DNN_Dropout
from dlnn.models.sda import SdA, Sda_xy
import theano.tensor as T
from dlnn.utils.network_config import NetworkConfig
from dlnn.utils.sda_config import SdAConfig
from dlnn.utils.utils import parse_arguments, save_two_integers, read_two_integers
from dlnn.learning.sgd import train_sgd, validate_by_minibatch
from dlnn.utils.utils import shared_dataset_X
from numpy import dtype, shape
from dlnn.factories import Sda_factory, Sda_xy_factory, DNN_factory, Parellel_Sda_factory
# In[6]:
from DL_libs import performance_score, Preprocessing_Scaler_with_mean_point5, cal_epochs, pretrain_a_Sda_with_estop, train_a_Sda, trainSda
#filename = 'SUCCESS_log_CrossValidation_load_DL_remoteFisherM1_DL_RE_US_DL_RE_US_1_1_19MAY2014.txt'
#filename = 'listOfDDIsHaveOver2InterfacesHave40-75_Examples_2010_real_selected.txt' #for testing

# set settings for this script
settings = {}
settings['filename'] = 'list_of_DDIs_for_contact_matrix_paper123.txt'
#settings['filename'] = 'testDDI.txt'
settings['fisher_mode'] = 'FisherM1ONLY'# settings['fisher_mode'] = 'FisherM1ONLY'
settings['with_auc_score'] = False
settings['n_cores'] = 8
settings['reduce_ratio'] = 1 
settings['SVM'] = 1
settings['DL'] = 1
settings['DL_old'] = 1
settings['Sda_new'] = 1
settings['DL_xy'] = 1
settings['SAE_SVM'] = 1
settings['SAE_SVM_COMBO'] = 0
settings['SVM_RBF'] = 1
settings['SAE_SVM_RBF'] = 0
settings['SAE_SVM_RBF_COMBO'] = 0
settings['SVM_POLY'] = 0
settings['DL_S'] = 1
settings['DL_U'] = 0
settings['DL_S_new']  = 1
settings['Sda_xy_with_first'] = 1
settings['DL_S_new_contraction'] = 1
settings['DL_S_new_sparsity'] = 1
settings['DL_S_new_weight_decay'] = 2
settings['DL_S_new_Drop_out'] = 1


settings['finetune_lr'] = 0.1
settings['batch_size'] = 100
settings['pretraining_interations'] = 20000
settings['pretrain_lr'] = 0.001
settings['training_interations'] = 20000
#settings['training_epochs'] = 10001
settings['hidden_layers_sizes'] = [100, 100]
settings['corruption_levels'] = [0, 0]

cfg = settings.copy()
cfg['learning_rate'] = 0.001  # this is for pretraining
#cfg['train-data'] = "/home/du/Dropbox/Project/libs/dlnn/cmds/train.pickle.gz,partition=600m"  
#cfg['batch_size'] = 100
cfg['wdir'] = '/home/du/Documents/tmp'
cfg['param-output-file']= 'sda.mdl'
cfg['sparsity'] = 0
cfg['sparsity_weight'] = 0
 # 1 means use equal weight for X. Y, 0 means use the scaled weight Y is set 1/size of X, None means use the original vector.
cfg['n_outs'] = 1 
#cfg['lrate'] ='C:0.08:5 ' #'D:1:0.5:0.05,0.005:5000'
#cfg['lrate'] = 'D:0.1:0.5:0.05,0.005:1500' #'D:0.1:0.5:0.05,0.005:1500'
cfg['lrate_pre'] ='D:0.1:0.5:0.05,0.005:'
'''
constant     --lrate="C:l:n"              
Eg. C:0.08:15                  
    run n iterations with lrate = l unchanged
newbob              
    --lrate="D:l:c:dh,ds:n"    
Eg. D:0.08:0.5:0.05,0.05:8              
    starts with the learning rate l; if the validation error reduction between two consecutive epochs is less than dh, the learning rate is scaled by c during each of the remaining epochs. Traing finally terminates when the validation error reduction between two consecutive epochs falls below ds. n is the minimum epoch number after which scaling can be performed.
min-rate newbob
    --lrate="MD:l:c:dh,ml:n"    
Eg. MD:0.08:0.5:0.05,0.0002:8
    the same as newbob, except that training terminates when the learning rate value falls below ml
fixed newbob
    --lrate="FD:l:c:eh,es"    
Eg. FD:0.08:0.5:10,6     starts with the learning rate l; after eh epochs, the learning rate starts to be scaled by c. Traing terminates when doing another es epochs after scaling starts. n is the minimum epoch number after which scaling can be performed.
'''
filename = settings['filename']
file_obj = FileOperator(filename)
ddis = file_obj.readStripLines()
import logging
import time
current_date = time.strftime("%m_%d_%Y")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logname = 'log_DL_contact_matrix_load' + current_date + '.log'
handler = logging.FileHandler(logname)
handler.setLevel(logging.DEBUG)

# create a logging format

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger

logger.addHandler(handler)

logger.info('Input DDI file: ' + filename)
#logger.debug('This message should go to the log file')
for key, value in cfg.items():
    logger.info(key +': '+ str(value))


# In[6]:




# In[7]:

class DDI_family_base(object):
    #def __init__(self, ddi, Vectors_Fishers_aaIndex_raw_folder = '/home/du/Documents/Vectors_Fishers_aaIndex_raw_2014/'):
    #def __init__(self, ddi, Vectors_Fishers_aaIndex_raw_folder = '/home/sun/Downloads/contactmatrix/contactmatrixanddeeplearningcode/data_test/'):
    #def __init__(self, ddi, Vectors_Fishers_aaIndex_raw_folder = '/home/du/Documents/Vectors_Fishers_aaIndex_raw_2014_paper/'):
    def __init__(self, ddi, Vectors_Fishers_aaIndex_raw_folder = '/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw_2014_paper/'):
        """ get total number of sequences in a ddi familgy
        Attributes:
            ddi: string ddi name
            Vectors_Fishers_aaIndex_raw_folder: string, folder
            total_number_of_sequences: int
            raw_data: dict raw_data[2]
        LOO_data['FisherM1'][1]

        """
        self.ddi = ddi
        self.Vectors_Fishers_aaIndex_raw_folder = Vectors_Fishers_aaIndex_raw_folder
        self.ddi_folder = self.Vectors_Fishers_aaIndex_raw_folder + ddi + '/'
        self.total_number_of_sequences = self.get_total_number_of_sequences()
        self.raw_data = {}
        self.positve_negative_number = {}
        self.equal_size_data = {}
        for seq_no in range(1, self.total_number_of_sequences+1):
            self.raw_data[seq_no] = self.get_raw_data_for_selected_seq(seq_no)
            try:
                #positive_file = self.ddi_folder + 'numPos_'+ str(seq_no) + '.txt'
                #file_obj = FileOperator(positive_file)
                #lines = file_obj.readStripLines()
                #import pdb; pdb.set_trace()
                count_pos = int(np.sum(self.raw_data[seq_no][:, -1]))
                count_neg = self.raw_data[seq_no].shape[0] - count_pos
                #self.positve_negative_number[seq_no] = {'numPos': int(float(lines[0]))}
                #assert int(float(lines[0])) == count_pos
                self.positve_negative_number[seq_no] = {'numPos': count_pos}
                #negative_file = self.ddi_folder + 'numNeg_'+ str(seq_no) + '.txt'
                #file_obj = FileOperator(negative_file)
                #lines = file_obj.readStripLines()
                #self.positve_negative_number[seq_no]['numNeg'] =  int(float(lines[0]))
                self.positve_negative_number[seq_no]['numNeg'] =  count_neg
            except Exception,e:
                print ddi, seq_no
                print str(e)
                logger.info(ddi + str(seq_no))
                logger.info(str(e))                
            # get data for equal positive and negative
            n_pos = self.positve_negative_number[seq_no]['numPos']
            n_neg = self.positve_negative_number[seq_no]['numNeg']
            index_neg = range(n_pos, n_pos + n_neg)
            random.shuffle(index_neg)
            index_neg = index_neg[: n_pos]
            positive_examples = self.raw_data[seq_no][ : n_pos, :]
            negative_examples = self.raw_data[seq_no][index_neg, :]
            self.equal_size_data[seq_no] = np.vstack((positive_examples, negative_examples))
    def get_LOO_training_and_reduced_traing(self, seq_no, fisher_mode = 'FisherM1ONLY' , reduce_ratio = 4):
        """ get the leave one out traing data, reduced traing
        Parameters:
            seq_no: 
            fisher_mode: default 'FisherM1ONLY'
        Returns:
            (train_X_LOO, train_y_LOO),(train_X_reduced, train_y_reduced),  (test_X, test_y)
        """
        train_X_LOO = np.array([])
        train_y_LOO = np.array([])
        train_X_reduced = np.array([])
        train_y_reduced = np.array([])

        total_number_of_sequences = self.total_number_of_sequences
        equal_size_data_selected_sequence = self.equal_size_data[seq_no]
        
        #get test data for selected sequence
        test_X, test_y = self.select_X_y(equal_size_data_selected_sequence, fisher_mode = fisher_mode)
        total_sequences = range(1, total_number_of_sequences+1)
        loo_sequences = [i for i in total_sequences if i != seq_no]
        number_of_reduced = len(loo_sequences)/reduce_ratio if len(loo_sequences)/reduce_ratio !=0 else 1
        random.shuffle(loo_sequences)
        reduced_sequences = loo_sequences[:number_of_reduced]

        #for loo data
        for current_no in loo_sequences:
            raw_current_data = self.equal_size_data[current_no]
            current_X, current_y = self.select_X_y(raw_current_data, fisher_mode = fisher_mode)
            if train_X_LOO.ndim ==1:
                train_X_LOO = current_X
            else:
                train_X_LOO = np.vstack((train_X_LOO, current_X))
            train_y_LOO = np.concatenate((train_y_LOO, current_y))

        #for reduced data
        for current_no in reduced_sequences:
            raw_current_data = self.equal_size_data[current_no]
            current_X, current_y = self.select_X_y(raw_current_data, fisher_mode = fisher_mode)
            if train_X_reduced.ndim ==1:
                train_X_reduced = current_X
            else:
                train_X_reduced = np.vstack((train_X_reduced, current_X))
            train_y_reduced = np.concatenate((train_y_reduced, current_y))                

        return (train_X_LOO, train_y_LOO),(train_X_reduced, train_y_reduced), (test_X, test_y)
        
    #def get_ten_fold_crossvalid_one_subset(self, start_subset, end_subset, fisher_mode = 'FisherM1ONLY' , reduce_ratio = 4):
    def get_ten_fold_crossvalid_one_subset(self, train_index, test_index, fisher_mode = 'FisherM1ONLY' , reduce_ratio = 4):
        """ get traing data, reduced traing data for 10-fold crossvalidation
        Parameters:
            start_subset: index of start of the testing data
            end_subset: index of end of the testing data
            fisher_mode: default 'FisherM1ONLY'
        Returns:
            (train_X_10fold, train_y_10fold),(train_X_reduced, train_y_reduced),  (test_X, test_y)
        """
        train_X_10fold = np.array([])
        train_y_10fold = np.array([])
        train_X_reduced = np.array([])
        train_y_reduced = np.array([])
        test_X = np.array([])
        test_y = np.array([])

        total_number_of_sequences = self.total_number_of_sequences
        
        #get test data for selected sequence
        #for current_no in range(start_subset, end_subset):
        for num in test_index:
            current_no = num + 1
            raw_current_data = self.equal_size_data[current_no]
            current_X, current_y = self.select_X_y(raw_current_data, fisher_mode = fisher_mode)
            if test_X.ndim ==1:
                test_X = current_X
            else:
                test_X = np.vstack((test_X, current_X))
            test_y = np.concatenate((test_y, current_y))
        
        #total_sequences = range(1, total_number_of_sequences+1)
        #ten_fold_sequences = [i for i in total_sequences if not(i in range(start_subset, end_subset))]
        #number_of_reduced = len(ten_fold_sequences)/reduce_ratio if len(ten_fold_sequences)/reduce_ratio !=0 else 1
        #random.shuffle(ten_fold_sequences)
        #reduced_sequences = ten_fold_sequences[:number_of_reduced]
        
        number_of_reduced = len(train_index)/reduce_ratio if len(train_index)/reduce_ratio !=0 else 1
        random.shuffle(train_index)
        reduced_sequences = train_index[:number_of_reduced]

        #for 10-fold cross-validation data
        #for current_no in ten_fold_sequences:
        for num in train_index:
            current_no = num + 1
            raw_current_data = self.equal_size_data[current_no]
            current_X, current_y = self.select_X_y(raw_current_data, fisher_mode = fisher_mode)
            if train_X_10fold.ndim ==1:
                train_X_10fold = current_X
            else:
                train_X_10fold = np.vstack((train_X_10fold, current_X))
            train_y_10fold = np.concatenate((train_y_10fold, current_y))

        #for reduced data
        for num in reduced_sequences:
            current_no = num + 1
            raw_current_data = self.equal_size_data[current_no]
            current_X, current_y = self.select_X_y(raw_current_data, fisher_mode = fisher_mode)
            if train_X_reduced.ndim ==1:
                train_X_reduced = current_X
            else:
                train_X_reduced = np.vstack((train_X_reduced, current_X))
            train_y_reduced = np.concatenate((train_y_reduced, current_y))                

        return (train_X_10fold, train_y_10fold),(train_X_reduced, train_y_reduced), (test_X, test_y)
        
    def get_total_number_of_sequences(self):
        """ get total number of sequences in a ddi familgy
        Parameters:
            ddi: string
            Vectors_Fishers_aaIndex_raw_folder: string
        Returns:
            n: int
        """
        folder_path = self.Vectors_Fishers_aaIndex_raw_folder + self.ddi + '/' 
        filename = folder_path +'allPairs.txt'
        all_pairs = np.loadtxt(filename)
        return len(all_pairs)

    def get_raw_data_for_selected_seq(self, seq_no):
        """ get raw data for selected seq no in a family
        Parameters:
            ddi: 
            seq_no: 
        Returns:
            data: raw data in the sequence file
        """
        folder_path = self.Vectors_Fishers_aaIndex_raw_folder + self.ddi + '/' 
        filename = folder_path + 'F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_'+ str(seq_no) + '.txt'
        data = np.loadtxt(filename, dtype = theano.config.floatX)
        return data
    def select_X_y(self, data, fisher_mode = ''):
        """ select subset from the raw input data set
        Parameters:
            data: data from matlab txt file
            fisher_mode: subset base on this Fisher of AAONLY...
        Returns:
            selected X,  y
        """
        y = data[:,-1] # get lable
        if fisher_mode == 'FisherM1': # fisher m1 plus AA index
            a = data[:, 20:227]
            b = data[:, 247:454]
            X = np.hstack((a,b))
        elif fisher_mode == 'FisherM1ONLY': 
            a = data[:, 20:40]
            b = data[:, 247:267]
            X = np.hstack((a,b))
        elif fisher_mode == 'AAONLY':
            a = data[:, 40:227]
            b = data[:, 267:454]
            X = np.hstack((a,b))
        else:
            raise('there is an error in mode')
        return X, y


# In[7]:




# In[7]:




# In[8]:

import sklearn.preprocessing

def saveAsCsv(with_auc_score, fname, score_dict, arguments): #new
    newfile = False
    if os.path.isfile('report_' + fname + '.csv'):
        pass
    else:
        newfile = True
    csvfile = open('report_' + fname + '.csv', 'a+')
    writer = csv.writer(csvfile)
    if newfile == True:
        if with_auc_score == False:
            writer.writerow(['DDI', 'no.', 'FisherMode', 'method', 'isTest']+ score_dict.keys()) #, 'AUC'])
        else:
            writer.writerow(['DDI', 'no.', 'FisherMode', 'method', 'isTest'] + score_dict.keys())
    for arg in arguments:        
        writer.writerow([i for i in arg])
    csvfile.close()

#for 10-fold cross validation

def ten_fold_crossvalid_performance_for_all(ddis):
    for ddi in ddis:
        try:
            process_one_ddi_tenfold(ddi)
        except Exception,e:
            print str(e)
            logger.debug("There is a error in this ddi: %s" % ddi)
            logger.info(str(e))
def process_one_ddi_tenfold(ddi):
    """A function to waste CPU cycles"""
    logger.info('DDI: %s' % ddi)
    try:
        one_ddi_family = {}
        one_ddi_family[ddi] = Ten_fold_crossvalid_performance_for_one_ddi(ddi)
        one_ddi_family[ddi].get_ten_fold_crossvalid_perfermance(settings=cfg)
    except Exception,e:
        print str(e)
        logger.debug("There is a error in this ddi: %s" % ddi)
        logger.info(str(e))
    return None
class Ten_fold_crossvalid_performance_for_one_ddi(object):
        """ get the performance of ddi families
        Attributes:
            ddi: string ddi name
            Vectors_Fishers_aaIndex_raw_folder: string, folder
            total_number_of_sequences: int
            raw_data: dict raw_data[2]

        """
        def __init__(self, ddi):
            self.ddi_obj = DDI_family_base(ddi)
            self.ddi = ddi
        def get_ten_fold_crossvalid_perfermance(self, settings = None):
            fisher_mode = settings['fisher_mode']
            analysis_scr = []
            with_auc_score = settings['with_auc_score']
            reduce_ratio = settings['reduce_ratio']
            #for seq_no in range(1, self.ddi_obj.total_number_of_sequences+1):
            #subset_size = math.floor(self.ddi_obj.total_number_of_sequences / 10.0)
            kf = KFold(self.ddi_obj.total_number_of_sequences, n_folds = 10, shuffle = True)
            #for subset_no in range(1, 11):
            for ((train_index, test_index),subset_no) in izip(kf,range(1,11)):
            #for train_index, test_index in kf;
                print("Subset:", subset_no)
                print("Train index: ", train_index)
                print("Test index: ", test_index)
                #logger.info('subset number: ' + str(subset_no))
                (train_X_10fold, train_y_10fold),(train_X_reduced, train_y_reduced), (test_X, test_y) = self.ddi_obj.get_ten_fold_crossvalid_one_subset(train_index, test_index, fisher_mode = fisher_mode, reduce_ratio = reduce_ratio)
                standard_scaler = preprocessing.StandardScaler().fit(train_X_reduced)
                scaled_train_X = standard_scaler.transform(train_X_reduced)
                scaled_test_X = standard_scaler.transform(test_X)
                
                if settings['SVM']:
                    print "SVM"                   
                    Linear_SVC = LinearSVC(C=1, penalty="l2")
                    Linear_SVC.fit(scaled_train_X, train_y_reduced)
                    predicted_test_y = Linear_SVC.predict(scaled_test_X)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                    predicted_train_y = Linear_SVC.predict(scaled_train_X)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))                    
                if settings['SVM_RBF']:
                    print "SVM_RBF"
                    L1_SVC_RBF_Selector = SVC(C=1, gamma=0.01, kernel='rbf').fit(scaled_train_X, train_y_reduced)

                    predicted_test_y = L1_SVC_RBF_Selector.predict(scaled_test_X)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM_RBF', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                    predicted_train_y = L1_SVC_RBF_Selector.predict(scaled_train_X)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM_RBF', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
                if settings['SVM_POLY']:
                    print "SVM_POLY"
                    L1_SVC_POLY_Selector = SVC(C=1, kernel='poly').fit(scaled_train_X, train_y_reduced)

                    predicted_test_y = L1_SVC_POLY_Selector.predict(scaled_test_X)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM_POLY', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                    predicted_train_y = L1_SVC_POLY_Selector.predict(scaled_train_X)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM_POLY', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
                
                min_max_scaler = Preprocessing_Scaler_with_mean_point5()
                X_train_pre_validation_minmax = min_max_scaler.fit(train_X_reduced)
                X_train_pre_validation_minmax = min_max_scaler.transform(train_X_reduced)
                x_test_minmax = min_max_scaler.transform(test_X)
                
                x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax = train_test_split(X_train_pre_validation_minmax, 
                                                                                                  train_y_reduced
                                                                    , test_size=0.4, random_state=42)
                finetune_lr = settings['finetune_lr']
                batch_size = settings['batch_size']
                pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                #pretrain_lr=0.001
                pretrain_lr = settings['pretrain_lr']
                training_epochs = cal_epochs(settings['training_interations'], x_train_minmax, batch_size = batch_size)
                hidden_layers_sizes= settings['hidden_layers_sizes']
                corruption_levels = settings['corruption_levels']
                settings['epoch_number'] = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                # deep xy autoencoders
                settings['lrate'] = settings['lrate_pre'] + str(training_epochs)
                settings['n_ins'] = x_train_minmax.shape[1]
                if settings['DL_xy']:
                    cfg = settings.copy()
                    cfg['weight_y'] = 0.1
                    print 'DL_xy'
                    train_x = x_train_minmax; train_y = y_train_minmax                    
                    sdaf = Sda_xy_factory(cfg)
                    sdaf.sda.pretraining(train_x, train_y) 
                    dnnf = DNN_factory(cfg) 
                    dnnf.dnn.load_pretrain_from_Sda(sdaf.sda)
                    dnnf.dnn.finetuning((x_train_minmax,  y_train_minmax),(x_validation_minmax, y_validation_minmax))
                    
                    training_predicted = dnnf.dnn.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_xy', isTest) + tuple(performance_score(y_train, training_predicted).values()))

                    test_predicted = dnnf.dnn.predict(x_test_minmax)
                    y_test = test_y
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_xy', isTest) + tuple(performance_score(y_test, test_predicted).values()))
                if settings['Sda_xy_with_first']: 
                    cfg = settings.copy()
                    cfg['weight_y'] = 0.1
                    cfg['firstlayer_xy'] = 1
                    print 'firstlayer_xy' 
                    train_x = x_train_minmax; train_y = y_train_minmax                    
                    sdaf = Sda_xy_factory(cfg)
                    sdaf.sda.pretraining(train_x, train_y) 
                    dnnf = DNN_factory(cfg) 
                    dnnf.dnn.load_pretrain_from_Sda(sdaf.sda)
                    dnnf.dnn.finetuning((x_train_minmax,  y_train_minmax),(x_validation_minmax, y_validation_minmax))
                    
                    training_predicted = dnnf.dnn.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'Sda_xy_with_first', isTest) + tuple(performance_score(y_train, training_predicted).values()))

                    test_predicted = dnnf.dnn.predict(x_test_minmax)
                    y_test = test_y
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'Sda_xy_with_first', isTest) + tuple(performance_score(y_test, test_predicted).values()))
                if settings['Sda_new']:
                    print 'Sda_new'
                    cfg = settings.copy()
                    train_x = x_train_minmax; train_y = y_train_minmax                    
                    cfg['n_ins'] = train_x.shape[1]
                    sdaf = Sda_factory(cfg)
                    sda = sdaf.sda.pretraining(train_x = train_x)
                    sdaf.dnn.finetuning((x_train_minmax,  y_train_minmax),(x_validation_minmax, y_validation_minmax))                    
                    training_predicted = sdaf.dnn.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'Sda_new', isTest) + tuple(performance_score(y_train, training_predicted).values()))

                    test_predicted = sdaf.dnn.predict(x_test_minmax)
                    y_test = test_y
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'Sda_new', isTest) + tuple(performance_score(y_test, test_predicted).values()))
                            
                #### new prepresentation
                x = X_train_pre_validation_minmax
                a_MAE_A = pretrain_a_Sda_with_estop(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                        hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                new_x_train_minmax_A =  a_MAE_A.transform(X_train_pre_validation_minmax)
                new_x_test_minmax_A =  a_MAE_A.transform(x_test_minmax)
                standard_scaler = preprocessing.StandardScaler().fit(new_x_train_minmax_A)
                new_x_train_scaled = standard_scaler.transform(new_x_train_minmax_A)
                new_x_test_scaled = standard_scaler.transform(new_x_test_minmax_A)
                new_x_train_combo = np.hstack((scaled_train_X, new_x_train_scaled))
                new_x_test_combo = np.hstack((scaled_test_X, new_x_test_scaled))
                
                
                if settings['SAE_SVM']: 
                    print 'SAE followed by SVM'

                    Linear_SVC = LinearSVC(C=1, penalty="l2")
                    Linear_SVC.fit(new_x_train_scaled, train_y_reduced)
                    predicted_test_y = Linear_SVC.predict(new_x_test_scaled)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new
                    predicted_train_y = Linear_SVC.predict(new_x_train_scaled)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
                if settings['SAE_SVM_RBF']: 
                    print 'SAE followed by SVM RBF'
                    x = X_train_pre_validation_minmax
                    L1_SVC_RBF_Selector = SVC(C=1, gamma=0.01, kernel='rbf').fit(new_x_train_scaled, train_y_reduced)
                    predicted_test_y = L1_SVC_RBF_Selector.predict(new_x_test_scaled)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM_RBF', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new
                    predicted_train_y = L1_SVC_RBF_Selector.predict(new_x_train_scaled)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM_RBF', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
                if settings['SAE_SVM_COMBO']: 
                    print 'SAE followed by SVM with combo feature'
                    Linear_SVC = LinearSVC(C=1, penalty="l2")
                    Linear_SVC.fit(new_x_train_combo, train_y_reduced)
                    predicted_test_y = Linear_SVC.predict(new_x_test_combo)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM_COMBO', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new
                    predicted_train_y = Linear_SVC.predict(new_x_train_combo)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM_COMBO', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))                                
                if settings['SAE_SVM_RBF_COMBO']: 
                    print 'SAE followed by SVM RBF with combo feature'
                    L1_SVC_RBF_Selector = SVC(C=1, gamma=0.01, kernel='rbf').fit(new_x_train_combo, train_y_reduced)
                    predicted_test_y = L1_SVC_RBF_Selector.predict(new_x_test_combo)        
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM_RBF_COMBO', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new
                    predicted_train_y = L1_SVC_RBF_Selector.predict(new_x_train_combo)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM_RBF_COMBO', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))                                                                  
                    
                if settings['DL']:
                    print "direct deep learning"
                    sda = train_a_Sda(x_train_minmax, pretrain_lr, finetune_lr,
                                      y_train_minmax,
                                 x_validation_minmax, y_validation_minmax , 
                                 x_test_minmax, test_y,
                                 hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                                 training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, n_outs = settings['n_outs'] 
                                 
                     )
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL', isTest) + tuple(performance_score(y_train, training_predicted).values()))

                    test_predicted = sda.predict(x_test_minmax)
                    y_test = test_y
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL', isTest) + tuple(performance_score(y_test, test_predicted).values()))
                if settings['DL_old']:
                    print "direct deep learning old without early stop"
                    sda = trainSda(x_train_minmax, pretrain_lr, finetune_lr,
                                      y_train_minmax,
                                 x_validation_minmax, y_validation_minmax , 
                                 x_test_minmax, test_y,
                                 hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                                 training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, n_outs = settings['n_outs'] 
                                 
                     )
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_old', isTest) + tuple(performance_score(y_train, training_predicted).values()))

                    test_predicted = sda.predict(x_test_minmax)
                    y_test = test_y
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_old', isTest) + tuple(performance_score(y_test, test_predicted).values()))                
                if settings['DL_U']:
                # deep learning using unlabeled data for pretraining
                    print 'deep learning with unlabel data'
                    pretraining_X_minmax = min_max_scaler.transform(train_X_10fold)
                    pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                    sda_unlabel = trainSda(x_train_minmax, y_train_minmax,
                                 x_validation_minmax, y_validation_minmax , 
                                 x_test_minmax, test_y, 
                                 pretraining_X_minmax = pretraining_X_minmax,
                                 hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                                 training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                                 pretrain_lr = pretrain_lr, finetune_lr=finetune_lr
                     )
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda_unlabel.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_U', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))

                    test_predicted = sda_unlabel.predict(x_test_minmax)
                    y_test = test_y

                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_U', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))
                if settings['DL_S']:
                    # deep learning using split network
                    y_test = test_y
                    print 'deep learning using split network'
                    # get the new representation for A set. first 784-D
                    pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                    
                    x = x_train_minmax[:, :x_train_minmax.shape[1]/2]
                    print "original shape for A", x.shape
                    a_MAE_A = pretrain_a_Sda_with_estop(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                            hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                    new_x_train_minmax_A =  a_MAE_A.transform(x_train_minmax[:, :x_train_minmax.shape[1]/2])
                    x = x_train_minmax[:, x_train_minmax.shape[1]/2:]
                    
                    print "original shape for B", x.shape
                    a_MAE_B = pretrain_a_Sda_with_estop(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                            hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                    new_x_train_minmax_B =  a_MAE_B.transform(x_train_minmax[:, x_train_minmax.shape[1]/2:])
                    
                    new_x_test_minmax_A = a_MAE_A.transform(x_test_minmax[:, :x_test_minmax.shape[1]/2])
                    new_x_test_minmax_B = a_MAE_B.transform(x_test_minmax[:, x_test_minmax.shape[1]/2:])
                    new_x_validation_minmax_A = a_MAE_A.transform(x_validation_minmax[:, :x_validation_minmax.shape[1]/2])
                    new_x_validation_minmax_B = a_MAE_B.transform(x_validation_minmax[:, x_validation_minmax.shape[1]/2:])
                    new_x_train_minmax_whole = np.hstack((new_x_train_minmax_A, new_x_train_minmax_B))
                    new_x_test_minmax_whole = np.hstack((new_x_test_minmax_A, new_x_test_minmax_B))
                    new_x_validationt_minmax_whole = np.hstack((new_x_validation_minmax_A, new_x_validation_minmax_B))

                    
                    sda_transformed = train_a_Sda(new_x_train_minmax_whole, pretrain_lr, finetune_lr,
                                                  y_train_minmax,
                         new_x_validationt_minmax_whole, y_validation_minmax , 
                         new_x_test_minmax_whole, y_test,
                         hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                         training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                         
                         )
                    
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda_transformed.predict(new_x_train_minmax_whole)
                    y_train = y_train_minmax
                    
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))

                    test_predicted = sda_transformed.predict(new_x_test_minmax_whole)
                    y_test = test_y

                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))
                if settings['DL_S_new']:
                    # deep learning using split network
                    print 'new deep learning using split network'

                    cfg = settings.copy()
                    p_sda = Parellel_Sda_factory(cfg)                    
                    p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                    
                    isTest = False #new
                    training_predicted = p_sda.predict(x_train_minmax)
                    y_train = y_train_minmax                                       
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))
                    
                    isTest = True #new
                    y_test = test_y
                    test_predicted = p_sda.predict(x_test_minmax)
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))            
                if settings['DL_S_new_contraction']:
                    print 'DL_S_new_contraction'
                    cfg = settings.copy()
                    cfg['contraction_level'] = 0.001
                    p_sda = Parellel_Sda_factory(cfg)                    
                    p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                    
                    isTest = False #new
                    training_predicted = p_sda.predict(x_train_minmax)
                    y_train = y_train_minmax                                       
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new_contraction', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))
                    
                    isTest = True #new
                    y_test = test_y
                    test_predicted = p_sda.predict(x_test_minmax)
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new_contraction', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))            
               
                if settings['DL_S_new_sparsity'] == 1:
                    print 'DL_S_new_sparsity'
                    cfg = settings.copy()
                    cfg['sparsity'] = 0.01
                    cfg['sparsity_weight'] = 0.01
                    p_sda = Parellel_Sda_factory(cfg)                    
                    p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                    
                    isTest = False #new
                    training_predicted = p_sda.predict(x_train_minmax)
                    y_train = y_train_minmax                                       
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new_sparsity', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))
                    
                    isTest = True #new
                    y_test = test_y
                    test_predicted = p_sda.predict(x_test_minmax)
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new_sparsity', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))            
               
                if settings['DL_S_new_weight_decay'] == 2:
                    cfg = settings.copy()
                    cfg['l2_reg'] =0.01
                    print 'l2_reg'
                    p_sda = Parellel_Sda_factory(cfg)                    
                    p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                    
                    isTest = False #new
                    training_predicted = p_sda.predict(x_train_minmax)
                    y_train = y_train_minmax                                       
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'l2_reg', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))
                    
                    isTest = True #new
                    y_test = test_y
                    test_predicted = p_sda.predict(x_test_minmax)
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'l2_reg', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))            
                    
                if settings['DL_S_new_weight_decay'] == 1:
                    print 'l1_reg'
                    cfg = settings.copy()
                    cfg['l1_reg'] =0.01 
                    p_sda = Parellel_Sda_factory(cfg)                    
                    p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                    
                    isTest = False #new
                    training_predicted = p_sda.predict(x_train_minmax)
                    y_train = y_train_minmax                                       
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'l1_reg', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))
                    
                    isTest = True #new
                    y_test = test_y
                    test_predicted = p_sda.predict(x_test_minmax)
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'l1_reg', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))            
                                     
                if settings['DL_S_new_Drop_out'] == 1:
                    
                    cfg = settings.copy()
                    cfg['dropout_factor'] = 0.3
                    print 'DL_S_new_Drop_out'
                    p_sda = Parellel_Sda_factory(cfg)                    
                    p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                    
                    isTest = False #new
                    training_predicted = p_sda.predict(x_train_minmax)
                    y_train = y_train_minmax                                       
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new_Drop_out', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))
                    
                    isTest = True #new
                    y_test = test_y
                    test_predicted = p_sda.predict(x_test_minmax)
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S_new_Drop_out', isTest) + tuple(performance_score(y_test, test_predicted, with_auc_score).values()))            
                                     
            report_name = filename + '_' + '_newDL_'.join(map(str, hidden_layers_sizes)) +                             '_' + str(pretrain_lr) + '_' + str(finetune_lr) + '_' + str(settings['training_interations']) + '_' + current_date
            saveAsCsv(with_auc_score, report_name, performance_score(test_y, predicted_test_y, with_auc_score), analysis_scr)


# In[10]:

#LOO_out_performance_for_all(ddis)
#LOO_out_performance_for_all(ddis)
#'''
from multiprocessing import Pool
pool = Pool(settings['n_cores'])
pool.map(process_one_ddi_tenfold, ddis[:20])
pool.close()
pool.join()
#'''
#process_one_ddi_tenfold(ddis[1])
# In[25]:

x = logging._handlers.copy()
for i in x:
    logger.removeHandler(i)
    i.flush()
    i.close()

