
# coding: utf-8

# In[3]:

import sys, os
sys.path.append('../../../libs/')
import os.path
import IO_class
from IO_class import FileOperator
from sklearn import cross_validation
import sklearn
import numpy as np
import csv
from dateutil import parser
from datetime import timedelta
from sklearn import svm
import numpy as np
import pandas as pd
import pdb
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn import preprocessing
import sklearn
import scipy.stats as ss
from sklearn.svm import LinearSVC
import random
from DL_libs import *
from itertools import izip #new
import math
from sklearn.svm import SVC


# In[4]:

#filename = 'SUCCESS_log_CrossValidation_load_DL_remoteFisherM1_DL_RE_US_DL_RE_US_1_1_19MAY2014.txt'
#filename = 'listOfDDIsHaveOver2InterfacesHave40-75_Examples_2010_real_selected.txt' #for testing

# set settings for this script
settings = {}
settings['filename'] = 'ddi_examples_40_60_over2top_diff_name_2014.txt'
settings['fisher_mode'] = 'FisherM1'
settings['predicted_score'] = False
settings['reduce_ratio'] = 8
settings['SVM'] = 1
settings['DL'] = 1
settings['SAE_SVM'] = 0
settings['SVM_RBF'] = 0
settings['DL_S'] = 0
settings['DL_U'] = 1

settings['finetune_lr'] = 1
settings['batch_size'] = 100
settings['pretraining_interations'] = 5008
settings['pretrain_lr'] = 0.001
settings['training_epochs'] = 1508
settings['hidden_layers_sizes'] = [100, 100]
settings['corruption_levels'] = [0,0]


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
for key, value in settings.items():
    logger.info(key +': '+ str(value))


# In[5]:

ddis


# In[28]:

class DDI_family_base(object):
    #def __init__(self, ddi, Vectors_Fishers_aaIndex_raw_folder = '/home/du/Documents/Vectors_Fishers_aaIndex_raw_2014/'):
    #def __init__(self, ddi, Vectors_Fishers_aaIndex_raw_folder = '/home/sun/Downloads/contactmatrix/contactmatrixanddeeplearningcode/data_test/'):
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
        data = np.loadtxt(filename)
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


# In[28]:




# In[29]:

import sklearn.preprocessing

def performance_score(target_label, predicted_label, predicted_score = False, print_report = True): 
    """ get performance matrix for prediction
        Attributes:
            target_label: int 0, 1
            predicted_label: 0, 1 or ranking
            predicted_score: bool if False, predicted_label is from 0, 1. If Ture, predicted_label is ranked, need to get AUC score.
            print_report: if True, print the perfromannce on screen
    """
    import sklearn
    from sklearn.metrics import roc_auc_score
    score = {}
    if predicted_score == False:
        score['accuracy'] = sklearn.metrics.accuracy_score(target_label, predicted_label)
        score['precision'] = sklearn.metrics.precision_score(target_label, predicted_label, pos_label=1)
        score['recall'] = sklearn.metrics.recall_score(target_label, predicted_label, pos_label=1)
    if predicted_score == True:
        auc_score  = roc_auc_score(target_label, predicted_label)
        score['auc_score'] = auc_score
        target_label = [x >= 0.5 for x in target_label]
        score['accuracy'] = sklearn.metrics.accuracy_score(target_label, predicted_label)
        score['precision'] = sklearn.metrics.precision_score(target_label, predicted_label, pos_label=1)
        score['recall'] = sklearn.metrics.recall_score(target_label, predicted_label, pos_label=1)
    if print_report == True:
        for key, value in score.iteritems():
            print key, '{percent:.1%}'.format(percent=value)
    return score

def saveAsCsv(predicted_score, fname, score_dict, *arguments): #new
    newfile = False
    if os.path.isfile('report_' + fname + '.csv'):
        pass
    else:
        newfile = True
    csvfile = open('report_' + fname + '.csv', 'a+')
    writer = csv.writer(csvfile)
    if newfile == True:
        if predicted_score == False:
            writer.writerow(['DDI', 'no.', 'FisherMode', 'method', 'isTest']+ score_dict.keys()) #, 'AUC'])
        else:
            writer.writerow(['DDI', 'no.', 'FisherMode', 'method', 'isTest'] + score_dict.keys())
    for arg in arguments:        
        writer.writerows(arg)
    csvfile.close()

def LOO_out_performance_for_all(ddis):
    for ddi in ddis:
        try:
            one_ddi_family = LOO_out_performance_for_one_ddi(ddi)
            one_ddi_family.get_LOO_perfermance(settings = settings)
        except Exception,e:
            print str(e)
            logger.info("There is a error in this ddi: %s" % ddi)
            logger.info(str(e))

        
class LOO_out_performance_for_one_ddi(object):
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

        def get_LOO_perfermance(self, settings = None):
            fisher_mode = settings['fisher_mode']
            analysis_scr = []
            predicted_score = settings['predicted_score'] 
            reduce_ratio = settings['reduce_ratio'] 
            for seq_no in range(1, self.ddi_obj.total_number_of_sequences+1):
                print seq_no
                logger.info('sequence number: ' + str(seq_no))
                if settings['SVM']:
                    print "SVM"
                    (train_X_LOO, train_y_LOO),(train_X_reduced, train_y_reduced), (test_X, test_y) = self.ddi_obj.get_LOO_training_and_reduced_traing(seq_no,fisher_mode = fisher_mode, reduce_ratio = reduce_ratio)
                    standard_scaler = preprocessing.StandardScaler().fit(train_X_reduced)
                    scaled_train_X = standard_scaler.transform(train_X_reduced)
                    scaled_test_X = standard_scaler.transform(test_X)
                    Linear_SVC = LinearSVC(C=1, penalty="l2")
                    Linear_SVC.fit(scaled_train_X, train_y_reduced)
                    predicted_test_y = Linear_SVC.predict(scaled_test_X)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'SVM', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                    predicted_train_y = Linear_SVC.predict(scaled_train_X)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'SVM', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
                # Deep learning part
                min_max_scaler = Preprocessing_Scaler_with_mean_point5()
                X_train_pre_validation_minmax = min_max_scaler.fit(train_X_reduced)
                X_train_pre_validation_minmax = min_max_scaler.transform(train_X_reduced)
                x_test_minmax = min_max_scaler.transform(test_X)
                pretraining_X_minmax = min_max_scaler.transform(train_X_LOO)
                x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax = train_test_split(X_train_pre_validation_minmax, 
                                                                                                  train_y_reduced
                                                                    , test_size=0.4, random_state=42)
                finetune_lr = settings['finetune_lr']
                batch_size = settings['batch_size']
                pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                #pretrain_lr=0.001
                pretrain_lr = settings['pretrain_lr']
                training_epochs = settings['training_epochs']
                hidden_layers_sizes= settings['hidden_layers_sizes']
                corruption_levels = settings['corruption_levels']
                if settings['DL']:
                    print "direct deep learning"
                    # direct deep learning 
                    sda = trainSda(x_train_minmax, y_train_minmax,
                                 x_validation_minmax, y_validation_minmax , 
                                 x_test_minmax, test_y,
                                 hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                                 training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                                 pretrain_lr = pretrain_lr, finetune_lr=finetune_lr
                     )
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'DL', isTest) + tuple(performance_score(y_train, training_predicted).values()))

                    test_predicted = sda.predict(x_test_minmax)
                    y_test = test_y
                    isTest = True; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'DL', isTest) + tuple(performance_score(y_test, test_predicted).values()))

                if 0:
                    # deep learning using unlabeled data for pretraining
                    print 'deep learning with unlabel data'
                    pretraining_epochs_for_reduced = cal_epochs(1500, pretraining_X_minmax, batch_size = batch_size)
                    sda_unlabel = trainSda(x_train_minmax, y_train_minmax,
                                 x_validation_minmax, y_validation_minmax , 
                                 x_test_minmax, test_y, 
                                 pretraining_X_minmax = pretraining_X_minmax,
                                 hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                                 training_epochs = training_epochs, pretraining_epochs = pretraining_epochs_for_reduced, 
                                 pretrain_lr = pretrain_lr, finetune_lr=finetune_lr
                     )
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda_unlabel.predict(x_train_minmax)
                    y_train = y_train_minmax
                    isTest = False; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'DL_U', isTest) + tuple(performance_score(y_train, training_predicted, predicted_score).values()))

                    test_predicted = sda_unlabel.predict(x_test_minmax)
                    y_test = test_y

                    isTest = True; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'DL_U', isTest) + tuple(performance_score(y_test, test_predicted, predicted_score).values()))
                if settings['Split_DL']:
                    # deep learning using split network
                    print 'deep learning using split network'
                    # get the new representation for A set. first 784-D
                    pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                    hidden_layers_sizes= settings['hidden_layers_sizes']
                    corruption_levels = settings['corruption_levels']
                    
                    x = x_train_minmax[:, :x_train_minmax.shape[1]/2]
                    print "original shape for A", x.shape
                    a_MAE_A = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                            hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                    new_x_train_minmax_A =  a_MAE_A.transform(x_train_minmax[:, :x_train_minmax.shape[1]/2])
                    x = x_train_minmax[:, x_train_minmax.shape[1]/2:]
                    
                    print "original shape for B", x.shape
                    a_MAE_B = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                            hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                    new_x_train_minmax_B =  a_MAE_B.transform(x_train_minmax[:, x_train_minmax.shape[1]/2:])
                    
                    new_x_test_minmax_A = a_MAE_A.transform(x_test_minmax[:, :x_test_minmax.shape[1]/2])
                    new_x_test_minmax_B = a_MAE_B.transform(x_test_minmax[:, x_test_minmax.shape[1]/2:])
                    new_x_validation_minmax_A = a_MAE_A.transform(x_validation_minmax[:, :x_validation_minmax.shape[1]/2])
                    new_x_validation_minmax_B = a_MAE_B.transform(x_validation_minmax[:, x_validation_minmax.shape[1]/2:])
                    new_x_train_minmax_whole = np.hstack((new_x_train_minmax_A, new_x_train_minmax_B))
                    new_x_test_minmax_whole = np.hstack((new_x_test_minmax_A, new_x_test_minmax_B))
                    new_x_validationt_minmax_whole = np.hstack((new_x_validation_minmax_A, new_x_validation_minmax_B))

                    finetune_lr = settings['finetune_lr']
                    batch_size = settings['batch_size']
                    pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                    #pretrain_lr=0.001
                    pretrain_lr = settings['pretrain_lr']
                    training_epochs = settings['training_epochs']
                    hidden_layers_sizes= settings['hidden_layers_sizes']
                    corruption_levels = settings['corruption_levels']
                    
                    sda_transformed = trainSda(new_x_train_minmax_whole, y_train_minmax,
                         new_x_validationt_minmax_whole, y_validation_minmax , 
                         new_x_test_minmax_whole, y_test,
                         hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                         training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                         pretrain_lr = pretrain_lr, finetune_lr=finetune_lr
                         )
                    
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda_transformed.predict(new_x_train_minmax_whole)
                    y_train = y_train_minmax
                    
                    isTest = False; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'DL_S', isTest) + tuple(performance_score(y_train, training_predicted, predicted_score).values()))

                    test_predicted = sda_transformed.predict(new_x_test_minmax_whole)
                    y_test = test_y

                    isTest = True; #new
                    analysis_scr.append((self.ddi, seq_no, fisher_mode, 'DL_S', isTest) + tuple(performance_score(y_test, test_predicted, predicted_score).values()))
            
            
            
            report_name = filename + '_' + '_'.join(map(str, hidden_layers_sizes)) +                             '_' + str(pretrain_lr) + '_' + str(finetune_lr) + '_' + str(reduce_ratio)+                             '_' +str(training_epochs) + '_' + current_date
            saveAsCsv(predicted_score, report_name, performance_score(y_test, test_predicted, predicted_score), analysis_scr)


# In[29]:




# In[30]:

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
    one_ddi_family = {}
    one_ddi_family[ddi] = Ten_fold_crossvalid_performance_for_one_ddi(ddi)
    one_ddi_family[ddi].get_ten_fold_crossvalid_perfermance(settings=settings)
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
            predicted_score = settings['predicted_score']
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
                if settings['SVM']:
                    print "SVM"
                    (train_X_10fold, train_y_10fold),(train_X_reduced, train_y_reduced), (test_X, test_y) = self.ddi_obj.get_ten_fold_crossvalid_one_subset(train_index, test_index, fisher_mode = fisher_mode, reduce_ratio = reduce_ratio)
                    standard_scaler = preprocessing.StandardScaler().fit(train_X_reduced)
                    scaled_train_X = standard_scaler.transform(train_X_reduced)
                    scaled_test_X = standard_scaler.transform(test_X)
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
                    standard_scaler = preprocessing.StandardScaler().fit(train_X_reduced)
                    scaled_train_X = standard_scaler.transform(train_X_reduced)
                    scaled_test_X = standard_scaler.transform(test_X)
                    L1_SVC_RBF_Selector = SVC(C=1, gamma=0.01, kernel='rbf').fit(scaled_train_X, train_y_reduced)

                    predicted_test_y = L1_SVC_RBF_Selector.predict(scaled_test_X)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM_RBF', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                    predicted_train_y = L1_SVC_RBF_Selector.predict(scaled_train_X)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SVM_RBF', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
          
                # direct deep learning 
                min_max_scaler = Preprocessing_Scaler_with_mean_point5()
                X_train_pre_validation_minmax = min_max_scaler.fit(train_X_reduced)
                X_train_pre_validation_minmax = min_max_scaler.transform(train_X_reduced)
                x_test_minmax = min_max_scaler.transform(test_X)
                pretraining_X_minmax = min_max_scaler.transform(train_X_10fold)
                x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax = train_test_split(X_train_pre_validation_minmax, 
                                                                                                  train_y_reduced
                                                                    , test_size=0.4, random_state=42)
                finetune_lr = settings['finetune_lr']
                batch_size = settings['batch_size']
                pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                #pretrain_lr=0.001
                pretrain_lr = settings['pretrain_lr']
                training_epochs = settings['training_epochs']
                hidden_layers_sizes= settings['hidden_layers_sizes']
                corruption_levels = settings['corruption_levels']
                if settings['SAE_SVM']: 
                    # SAE_SVM
                    print 'SAE followed by SVM'
                    x = X_train_pre_validation_minmax
                    a_MAE_A = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                            hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                    new_x_train_minmax_A =  a_MAE_A.transform(X_train_pre_validation_minmax)
                    new_x_test_minmax_A =  a_MAE_A.transform(x_test_minmax)
                    Linear_SVC = LinearSVC(C=1, penalty="l2")
                    Linear_SVC.fit(new_x_train_minmax_A, train_y_reduced)
                    predicted_test_y = Linear_SVC.predict(new_x_test_minmax_A)
                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                    predicted_train_y = Linear_SVC.predict(new_x_train_minmax_A)
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'SAE_SVM', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
                 
                                  
                    
                if settings['DL']:
                    print "direct deep learning"
                    sda = trainSda(x_train_minmax, y_train_minmax,
                                 x_validation_minmax, y_validation_minmax , 
                                 x_test_minmax, test_y,
                                 hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                                 training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                                 pretrain_lr = pretrain_lr, finetune_lr=finetune_lr
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

                if settings['DL_U']:
                # deep learning using unlabeled data for pretraining
                    print 'deep learning with unlabel data'
                    
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
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_U', isTest) + tuple(performance_score(y_train, training_predicted, predicted_score).values()))

                    test_predicted = sda_unlabel.predict(x_test_minmax)
                    y_test = test_y

                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_U', isTest) + tuple(performance_score(y_test, test_predicted, predicted_score).values()))
                if settings['DL_S']:
                    # deep learning using split network
                    print 'deep learning using split network'
                    # get the new representation for A set. first 784-D
                    pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
                    
                    x = x_train_minmax[:, :x_train_minmax.shape[1]/2]
                    print "original shape for A", x.shape
                    a_MAE_A = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                            hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                    new_x_train_minmax_A =  a_MAE_A.transform(x_train_minmax[:, :x_train_minmax.shape[1]/2])
                    x = x_train_minmax[:, x_train_minmax.shape[1]/2:]
                    
                    print "original shape for B", x.shape
                    a_MAE_B = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                            hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
                    new_x_train_minmax_B =  a_MAE_B.transform(x_train_minmax[:, x_train_minmax.shape[1]/2:])
                    
                    new_x_test_minmax_A = a_MAE_A.transform(x_test_minmax[:, :x_test_minmax.shape[1]/2])
                    new_x_test_minmax_B = a_MAE_B.transform(x_test_minmax[:, x_test_minmax.shape[1]/2:])
                    new_x_validation_minmax_A = a_MAE_A.transform(x_validation_minmax[:, :x_validation_minmax.shape[1]/2])
                    new_x_validation_minmax_B = a_MAE_B.transform(x_validation_minmax[:, x_validation_minmax.shape[1]/2:])
                    new_x_train_minmax_whole = np.hstack((new_x_train_minmax_A, new_x_train_minmax_B))
                    new_x_test_minmax_whole = np.hstack((new_x_test_minmax_A, new_x_test_minmax_B))
                    new_x_validationt_minmax_whole = np.hstack((new_x_validation_minmax_A, new_x_validation_minmax_B))

                    
                    sda_transformed = trainSda(new_x_train_minmax_whole, y_train_minmax,
                         new_x_validationt_minmax_whole, y_validation_minmax , 
                         new_x_test_minmax_whole, y_test,
                         hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                         training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                         pretrain_lr = pretrain_lr, finetune_lr=finetune_lr
                         )
                    
                    print 'hidden_layers_sizes:', hidden_layers_sizes
                    print 'corruption_levels:', corruption_levels
                    training_predicted = sda_transformed.predict(new_x_train_minmax_whole)
                    y_train = y_train_minmax
                    
                    isTest = False; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S', isTest) + tuple(performance_score(y_train, training_predicted, predicted_score).values()))

                    test_predicted = sda_transformed.predict(new_x_test_minmax_whole)
                    y_test = test_y

                    isTest = True; #new
                    analysis_scr.append((self.ddi, subset_no, fisher_mode, 'DL_S', isTest) + tuple(performance_score(y_test, test_predicted, predicted_score).values()))
            
            
            report_name = filename + '_' + '_test10fold_'.join(map(str, hidden_layers_sizes)) +                             '_' + str(pretrain_lr) + '_' + str(finetune_lr) + '_' + str(reduce_ratio)+                     '_' + str(training_epochs) + '_' + current_date
            saveAsCsv(predicted_score, report_name, performance_score(y_test, test_predicted, predicted_score), analysis_scr)


# In[1]:

ten_fold_crossvalid_performance_for_all(ddis[:])


# In[ ]:

#LOO_out_performance_for_all(ddis)


# In[25]:

x = logging._handlers.copy()
for i in x:
    log.removeHandler(i)
    i.flush()
    i.close()


# In[ ]:



