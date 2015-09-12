
# coding: utf-8

# In[1]:

# this part imports libs and load data from csv file
import sys
sys.path.append('../../../libs/')
import csv
from dateutil import parser
from datetime import timedelta
from sklearn import svm
import numpy as np
#import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import sklearn
import scipy.stats as ss
import cPickle
import gzip
import os
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
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
import pdb, PIL
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


# In[2]:

# set settings for this script
settings = {}
settings['with_auc_score'] = False
settings['reduce_ratio'] = 1
settings['SVM'] = 1
settings['SVM_RBF'] = 1
settings['SVM_POLY'] = 1
settings['DL'] = 1
settings['Log'] = 1
settings['SAE_Log'] = 0
settings['SAE_SVM'] = 0
settings['SAE_SVM_RBF'] = 0
settings['SAE_SVM_POLY'] = 0

settings['DL_S'] = 1
settings['SAE_S_SVM'] = 1
settings['SAE_S_SVM_RBF'] = 1
settings['SAE_S_SVM_POLY'] = 0



settings['number_iterations'] = 1


settings['finetune_lr'] = 1
settings['batch_size'] = 30
settings['pretraining_interations'] = 20000#10000
settings['pretrain_lr'] = 0.001
#settings['training_epochs'] = 300 #300
settings['training_interations'] = 20002 #300
settings['hidden_layers_sizes'] = [200, 200]
settings['corruption_levels'] = [0, 0]
settings['number_of_training'] = [10000]#[1000, 2500, 5000, 7500, 10000] # use all examples
settings['test_set_from_test'] = True

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
cfg['lrate_pre'] ='D:1:0.5:0.05,0.005:'
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
import logging
import time
current_date = time.strftime("%m_%d_%Y")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logname = 'log_DL_handwritten_digits' + current_date + '.log'
handler = logging.FileHandler(logname)
handler.setLevel(logging.DEBUG)

# create a logging format

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger

logger.addHandler(handler)

#logger.debug('This message should go to the log file')
for key, value in settings.items():
    logger.info(key +': '+ str(value))


# In[3]:

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
X_train,y_train = train_set
X_valid,y_valid = valid_set
X_total=np.vstack((X_train, X_valid))
X_total = np.array(X_total, dtype= theano.config.floatX)
print'sample size', X_total.shape
y_total = np.concatenate([y_train, y_valid])


# In[5]:

################## generate data from training set###################
array_A =[]
array_B =[]
for i in range(100000):
    array_A.append(np.random.random_integers(0, 59999))
    array_B.append(np.random.random_integers(0, 59999))
pos_index = []
neg_index = []
for index in xrange(100000):
    if y_total[array_A[index]] - y_total[array_B[index]] == 1:
        pos_index.append(index)
    else:
        neg_index.append(index)
print 'number of positive examples', len(pos_index)
selected_neg_index= neg_index[ : len(pos_index)]    

array_A = np.array(array_A)
array_B = np.array(array_B)
index_for_positive_image_A = array_A[pos_index]
index_for_positive_image_B = array_B[pos_index]
index_for_neg_image_A = array_A[selected_neg_index]
index_for_neg_image_B = array_B[selected_neg_index]

X_pos_A = X_total[index_for_positive_image_A]
X_pos_B = X_total[index_for_positive_image_B]
X_pos_whole = np.hstack((X_pos_A,X_pos_B))
X_neg_A = X_total[index_for_neg_image_A]
X_neg_B = X_total[index_for_neg_image_B]
X_neg_whole = np.hstack((X_neg_A, X_neg_B))
print X_pos_A.shape,  X_pos_B.shape, X_pos_whole.shape
print X_neg_A.shape,  X_neg_B.shape, X_neg_whole.shape

X_whole = np.vstack((X_pos_whole, X_neg_whole))
print X_whole.shape
y_pos = np.ones(X_pos_whole.shape[0])
y_neg = np.zeros(X_neg_whole.shape[0])
y_whole = np.concatenate([y_pos,y_neg])
print y_whole


# In[7]:

#pylab.imshow(imageB.reshape(28, 28), cmap="Greys")


# In[8]:


def saveAsCsv(with_auc_score, fname, score_dict, arguments): #new
    newfile = False
    if os.path.isfile('report_' + fname + '.csv'):
        pass
    else:
        newfile = True
    csvfile = open('report_' + fname + '.csv', 'a+')
    writer = csv.writer(csvfile)
    if newfile == True:
        writer.writerow(['no.', 'number_of_training', 'method', 'isTest']+ score_dict.keys()) #, 'AUC'])

    for arg in arguments:        
        writer.writerow([i for i in arg])
    csvfile.close()
def run_models(settings = None):
    analysis_scr = []
    with_auc_score = settings['with_auc_score']
    n_outs = settings['n_outs']
    for subset_no in xrange(1,settings['number_iterations']+1):
        print("Subset:", subset_no)
        
        ################## generate data ###################
        array_A =[]
        array_B =[]
        for i in range(100000):
            array_A.append(np.random.random_integers(0, 59999))
            array_B.append(np.random.random_integers(0, 59999))
        pos_index = []
        neg_index = []
        for index in xrange(100000):
            if y_total[array_A[index]] - y_total[array_B[index]] == 1:
                pos_index.append(index)
            else:
                neg_index.append(index)
        print 'number of positive examples', len(pos_index)
        selected_neg_index= neg_index[ : len(pos_index)]
        
        array_A = np.array(array_A)
        array_B = np.array(array_B)
        index_for_positive_image_A = array_A[pos_index]
        index_for_positive_image_B = array_B[pos_index]
        index_for_neg_image_A = array_A[selected_neg_index]
        index_for_neg_image_B = array_B[selected_neg_index]

        X_pos_A = X_total[index_for_positive_image_A]
        X_pos_B = X_total[index_for_positive_image_B]
        X_pos_whole = np.hstack((X_pos_A,X_pos_B))
        X_neg_A = X_total[index_for_neg_image_A]
        X_neg_B = X_total[index_for_neg_image_B]
        X_neg_whole = np.hstack((X_neg_A, X_neg_B))
        print X_pos_A.shape,  X_pos_B.shape, X_pos_whole.shape
        print X_neg_A.shape,  X_neg_B.shape, X_neg_whole.shape

        X_whole = np.vstack((X_pos_whole, X_neg_whole))
        print X_whole.shape
        y_pos = np.ones(X_pos_whole.shape[0])
        y_neg = np.zeros(X_neg_whole.shape[0])
        y_whole = np.concatenate([y_pos,y_neg])
        print y_whole.shape
        
        x_train_pre_validation, x_test, y_train_pre_validation, y_test = train_test_split(X_whole,y_whole,                                                            test_size=0.2, random_state=211)
        for number_of_training in settings['number_of_training']:
            
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_pre_validation[:number_of_training],
                                                                                                        y_train_pre_validation[:number_of_training],\
                                                                        test_size=0.2, random_state=21)
            '''
            x_train, x_validation, y_train, y_validation = train_test_split(x_train_pre_validation[:],
                                                                                                        y_train_pre_validation[:],\
                                                                        test_size=0.4, random_state=21)
            '''
            print x_train.shape, y_train.shape, x_validation.shape, \
            y_validation.shape, x_test.shape, y_test.shape
            x_train_minmax, x_validation_minmax, x_test_minmax = x_train, x_validation, x_test 
            train_X_reduced = x_train
            train_y_reduced = y_train
            test_X = x_test
            test_y = y_test
            y_train_minmax = y_train
            y_validation_minmax = y_validation
            y_test_minmax = y_test
            ###original data###
            ################ end of data ####################
            standard_scaler = preprocessing.StandardScaler().fit(train_X_reduced)
            scaled_train_X = standard_scaler.transform(train_X_reduced)
            scaled_test_X = standard_scaler.transform(test_X)
            if settings['SVM']:
                print "SVM"                   
                Linear_SVC = LinearSVC(C=1, penalty="l2")
                Linear_SVC.fit(scaled_train_X, y_train)
                predicted_test_y = Linear_SVC.predict(scaled_test_X)
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training, 'SVM', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                predicted_train_y = Linear_SVC.predict(scaled_train_X)
                isTest = False; #new
                analysis_scr.append(( subset_no,number_of_training, 'SVM', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))

            if settings['SVM_RBF']:
                print "SVM_RBF"
                L1_SVC_RBF_Selector = SVC(C=1, gamma=0.01, kernel='rbf').fit(scaled_train_X, y_train)
                predicted_test_y = L1_SVC_RBF_Selector.predict(scaled_test_X)
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training, 'SVM_RBF', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new
                predicted_train_y = L1_SVC_RBF_Selector.predict(scaled_train_X)
                isTest = False; #new
                analysis_scr.append((subset_no,number_of_training,  'SVM_RBF', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
                
            if settings['SVM_POLY']:
                print "SVM_POLY"
                L1_SVC_POLY_Selector = SVC(C=1, kernel='poly').fit(scaled_train_X, train_y_reduced)

                predicted_test_y = L1_SVC_POLY_Selector.predict(scaled_test_X)
                isTest = True; #new
                analysis_scr.append(( subset_no, number_of_training,'SVM_POLY', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                predicted_train_y = L1_SVC_POLY_Selector.predict(scaled_train_X)
                isTest = False; #new
                analysis_scr.append((subset_no, number_of_training,'SVM_POLY', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))

            if settings['Log']:
                print "Log"
                log_clf_l2 = sklearn.linear_model.LogisticRegression(C=1, penalty='l2')
                log_clf_l2.fit(scaled_train_X, train_y_reduced)
                predicted_test_y = log_clf_l2.predict(scaled_test_X)
                isTest = True; #new
                analysis_scr.append((subset_no,number_of_training, 'Log', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new
                predicted_train_y = log_clf_l2.predict(scaled_train_X)
                isTest = False; #new
                analysis_scr.append((subset_no, number_of_training,'Log', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))        

            # direct deep learning 

            finetune_lr = settings['finetune_lr']
            batch_size = settings['batch_size']
            pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
            #pretrain_lr=0.001
            pretrain_lr = settings['pretrain_lr']
            training_epochs = cal_epochs(settings['training_interations'], x_train_minmax, batch_size = batch_size)
            hidden_layers_sizes = settings['hidden_layers_sizes']
            corruption_levels = settings['corruption_levels']
            settings['lrate'] = settings['lrate_pre'] + str(training_epochs)
            
            if settings['DL']:
                print "direct deep learning"
                sda = trainSda(x_train_minmax, y_train,
                             x_validation_minmax, y_validation, 
                             x_test_minmax, test_y,
                             hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                             training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                             pretrain_lr = pretrain_lr, finetune_lr=finetune_lr, n_outs = n_outs
                 )
                print 'hidden_layers_sizes:', hidden_layers_sizes
                print 'corruption_levels:', corruption_levels
                test_predicted = sda.predict(x_test_minmax)
                isTest = True; #new
                analysis_scr.append((subset_no,number_of_training, 'DL', isTest) + tuple(performance_score(y_test, test_predicted).values()))
                training_predicted = sda.predict(x_train_minmax)
                isTest = False; #new
                analysis_scr.append((subset_no,number_of_training, 'DL', isTest) + tuple(performance_score(y_train, training_predicted).values()))

            ####transformed original data####    
            x = train_X_reduced
            a_MAE_original = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                    hidden_layers_sizes =hidden_layers_sizes, corruption_levels=corruption_levels)
            new_x_train_minmax_A =  a_MAE_original.transform(train_X_reduced)
            new_x_test_minmax_A =  a_MAE_original.transform(x_test_minmax)  
            standard_scaler = preprocessing.StandardScaler().fit(new_x_train_minmax_A)
            new_x_train_scaled = standard_scaler.transform(new_x_train_minmax_A)
            new_x_test_scaled = standard_scaler.transform(new_x_test_minmax_A)
            new_x_train_combo = np.hstack((scaled_train_X, new_x_train_scaled))
            new_x_test_combo = np.hstack((scaled_test_X, new_x_test_scaled))

            if settings['SAE_SVM']: 
                # SAE_SVM
                print 'SAE followed by SVM'

                Linear_SVC = LinearSVC(C=1, penalty="l2")
                Linear_SVC.fit(new_x_train_scaled, train_y_reduced)
                predicted_test_y = Linear_SVC.predict(new_x_test_scaled)
                isTest = True; #new
                analysis_scr.append(( subset_no, number_of_training,'SAE_SVM', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                predicted_train_y = Linear_SVC.predict(new_x_train_scaled)
                isTest = False; #new
                analysis_scr.append(( subset_no, number_of_training,'SAE_SVM', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
            if settings['SAE_Log']:
                print 'SAE followed by Log'
                log_clf_l2 = sklearn.linear_model.LogisticRegression(C=1, penalty='l2')
                log_clf_l2.fit(new_x_train_scaled, train_y_reduced)
                predicted_test_y = log_clf_l2.predict(new_x_test_scaled)
                isTest = True; #new
                analysis_scr.append((subset_no,number_of_training, 'SAE_Log', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new
                predicted_train_y = log_clf_l2.predict(new_x_train_scaled)
                isTest = False; #new
                analysis_scr.append((subset_no, number_of_training,'SAE_Log', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))        

            if settings['SAE_SVM_RBF']: 
                # SAE_SVM
                print 'SAE followed by SVM RBF'
                L1_SVC_RBF_Selector = SVC(C=1, gamma=0.01, kernel='rbf').fit(new_x_train_scaled, train_y_reduced)

                predicted_test_y = L1_SVC_RBF_Selector.predict(new_x_test_scaled)
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training, 'SAE_SVM_RBF', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                predicted_train_y = L1_SVC_RBF_Selector.predict(new_x_train_scaled)
                isTest = False; #new
                analysis_scr.append((subset_no, number_of_training, 'SAE_SVM_RBF', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))
            if settings['SAE_SVM_POLY']: 
                # SAE_SVM
                print 'SAE followed by SVM POLY'
                L1_SVC_RBF_Selector = SVC(C=1, kernel='poly').fit(new_x_train_scaled, train_y_reduced)

                predicted_test_y = L1_SVC_RBF_Selector.predict(new_x_test_scaled)
                isTest = True; #new
                analysis_scr.append((subset_no,  number_of_training,'SAE_SVM_POLY', isTest) + tuple(performance_score(test_y, predicted_test_y).values())) #new

                predicted_train_y = L1_SVC_RBF_Selector.predict(new_x_train_scaled)
                isTest = False; #new
                analysis_scr.append((subset_no, number_of_training, 'SAE_SVM_POLY', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y).values()))

            #### separated transformed data ####
            y_test = test_y
            print 'deep learning using split network'
            # get the new representation for A set. first 784-D
            pretraining_epochs = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)

            x = x_train_minmax[:, :x_train_minmax.shape[1]/2]
            print "original shape for A", x.shape
            a_MAE_A = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                    hidden_layers_sizes = [x/2 for x in hidden_layers_sizes], corruption_levels=corruption_levels)
            new_x_train_minmax_A =  a_MAE_A.transform(x_train_minmax[:, :x_train_minmax.shape[1]/2])
            x = x_train_minmax[:, x_train_minmax.shape[1]/2:]

            print "original shape for B", x.shape
            a_MAE_B = train_a_MultipleAEs(x, pretraining_epochs=pretraining_epochs, pretrain_lr=pretrain_lr, batch_size=batch_size, 
                                    hidden_layers_sizes = [x/2 for x in hidden_layers_sizes], corruption_levels=corruption_levels)
            new_x_train_minmax_B =  a_MAE_B.transform(x_train_minmax[:, x_train_minmax.shape[1]/2:])

            new_x_test_minmax_A = a_MAE_A.transform(x_test_minmax[:, :x_test_minmax.shape[1]/2])
            new_x_test_minmax_B = a_MAE_B.transform(x_test_minmax[:, x_test_minmax.shape[1]/2:])
            new_x_validation_minmax_A = a_MAE_A.transform(x_validation_minmax[:, :x_validation_minmax.shape[1]/2])
            new_x_validation_minmax_B = a_MAE_B.transform(x_validation_minmax[:, x_validation_minmax.shape[1]/2:])
            new_x_train_minmax_whole = np.hstack((new_x_train_minmax_A, new_x_train_minmax_B))
            new_x_test_minmax_whole = np.hstack((new_x_test_minmax_A, new_x_test_minmax_B))
            new_x_validationt_minmax_whole = np.hstack((new_x_validation_minmax_A, new_x_validation_minmax_B)) 
            standard_scaler = preprocessing.StandardScaler().fit(new_x_train_minmax_whole)
            new_x_train_minmax_whole_scaled = standard_scaler.transform(new_x_train_minmax_whole)
            new_x_test_minmax_whole_scaled = standard_scaler.transform(new_x_test_minmax_whole)            
            if settings['DL_S']:
                # deep learning using split network
                sda_transformed = trainSda(new_x_train_minmax_whole, y_train,
                     new_x_validationt_minmax_whole, y_validation , 
                     new_x_test_minmax_whole, y_test,
                     hidden_layers_sizes = hidden_layers_sizes, corruption_levels = corruption_levels, batch_size = batch_size , \
                     training_epochs = training_epochs, pretraining_epochs = pretraining_epochs, 
                     pretrain_lr = pretrain_lr, finetune_lr=finetune_lr
                     )
                print 'hidden_layers_sizes:', hidden_layers_sizes
                print 'corruption_levels:', corruption_levels

                predicted_test_y = sda_transformed.predict(new_x_test_minmax_whole)
                y_test = test_y
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training,'DL_S', isTest) + tuple(performance_score(y_test, predicted_test_y, with_auc_score).values()))

                training_predicted = sda_transformed.predict(new_x_train_minmax_whole)
                isTest = False; #new
                analysis_scr.append((subset_no,number_of_training, 'DL_S', isTest) + tuple(performance_score(y_train, training_predicted, with_auc_score).values()))
            if settings['SAE_S_SVM']:
                print 'SAE_S followed by SVM'

                Linear_SVC = LinearSVC(C=1, penalty="l2")
                Linear_SVC.fit(new_x_train_minmax_whole_scaled, train_y_reduced)
                predicted_test_y = Linear_SVC.predict(new_x_test_minmax_whole_scaled)
                isTest = True; #new
                analysis_scr.append(( subset_no, number_of_training,'SAE_S_SVM', isTest) + tuple(performance_score(test_y, predicted_test_y, with_auc_score).values())) #new

                predicted_train_y = Linear_SVC.predict(new_x_train_minmax_whole_scaled)
                isTest = False; #new
                analysis_scr.append(( subset_no,number_of_training, 'SAE_S_SVM', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y, with_auc_score).values()))
            if settings['SAE_S_SVM_RBF']: 
                print 'SAE S followed by SVM RBF'
                L1_SVC_RBF_Selector = SVC(C=1, gamma=0.01, kernel='rbf').fit(new_x_train_minmax_whole_scaled, train_y_reduced)

                predicted_test_y = L1_SVC_RBF_Selector.predict(new_x_test_minmax_whole_scaled)
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training, 'SAE_S_SVM_RBF', isTest) + tuple(performance_score(test_y, predicted_test_y, with_auc_score).values())) #new

                predicted_train_y = L1_SVC_RBF_Selector.predict(new_x_train_minmax_whole_scaled)
                isTest = False; #new
                analysis_scr.append((subset_no,  number_of_training,'SAE_S_SVM_RBF', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y, with_auc_score).values()))
            if settings['SAE_S_SVM_POLY']: 
                # SAE_SVM
                print 'SAE S followed by SVM POLY'
                L1_SVC_RBF_Selector = SVC(C=1, kernel='poly').fit(new_x_train_minmax_whole_scaled, train_y_reduced)

                predicted_test_y = L1_SVC_RBF_Selector.predict(new_x_test_minmax_whole_scaled)
                isTest = True; #new
                analysis_scr.append((subset_no,  number_of_training,'SAE_S_SVM_POLY', isTest) + tuple(performance_score(test_y, predicted_test_y, with_auc_score).values())) #new

                predicted_train_y = L1_SVC_RBF_Selector.predict(new_x_train_minmax_whole_scaled)
                isTest = False; #new
                analysis_scr.append((subset_no,  number_of_training,'SAE_S_SVM_POLY', isTest) + tuple(performance_score(train_y_reduced, predicted_train_y, with_auc_score).values()))

            settings['epoch_number'] = cal_epochs(settings['pretraining_interations'], x_train_minmax, batch_size = batch_size)
            # deep xy autoencoders
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
                analysis_scr.append((subset_no, number_of_training, 'DL_xy', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values()))

                test_predicted = dnnf.dnn.predict(x_test_minmax)
                y_test = test_y
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training, 'DL_xy', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))
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
                analysis_scr.append((subset_no, number_of_training, 'Sda_xy_with_first', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values()))
                test_predicted = dnnf.dnn.predict(x_test_minmax)
                y_test = test_y
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training, 'Sda_xy_with_first', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))
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
                analysis_scr.append((subset_no, number_of_training, 'Sda_new', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values()))

                test_predicted = sdaf.dnn.predict(x_test_minmax)
                y_test = test_y
                isTest = True; #new
                analysis_scr.append((subset_no, number_of_training, 'Sda_new', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))

            if settings['DL_S_new']:
                # deep learning using split network
                print 'new deep learning using split network'

                cfg = settings.copy()
                p_sda = Parellel_Sda_factory(cfg)                    
                p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                
                isTest = False #new
                training_predicted = p_sda.predict(x_train_minmax)
                y_train = y_train_minmax                                       
    
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values()))
                isTest = True #new
                y_test = test_y
                test_predicted = p_sda.predict(x_test_minmax)
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))            
            if settings['DL_S_new_contraction']:
                print 'DL_S_new_contraction'
                cfg = settings.copy()
                cfg['contraction_level'] = 0.001
                p_sda = Parellel_Sda_factory(cfg)                    
                p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                
                isTest = False #new
                training_predicted = p_sda.predict(x_train_minmax)
                y_train = y_train_minmax                                       
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new_contraction', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values()))
                isTest = True #new
                y_test = test_y
                test_predicted = p_sda.predict(x_test_minmax)
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new_contraction', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))   
            
            if settings['DL_S_new_sparsity'] == 1:
                print 'DL_S_new_sparsity'
                cfg = settings.copy()
                cfg['sparsity'] = 0.01
                cfg['sparsity_weight'] = 0.001
                p_sda = Parellel_Sda_factory(cfg)                    
                p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                
                isTest = False #new
                training_predicted = p_sda.predict(x_train_minmax)
                y_train = y_train_minmax                                       
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new_sparsity', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values()))
                isTest = True #new
                y_test = test_y
                test_predicted = p_sda.predict(x_test_minmax)           
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new_sparsity', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))            
            if settings['DL_S_new_weight_decay'] == 2:
                cfg = settings.copy()
                cfg['l2_reg'] =0.001
                print 'l2_reg'
                p_sda = Parellel_Sda_factory(cfg)                    
                p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                
                isTest = False #new
                training_predicted = p_sda.predict(x_train_minmax)
                y_train = y_train_minmax                                       
                analysis_scr.append((subset_no, number_of_training, 'l2_reg', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values()))                
                isTest = True #new
                y_test = test_y
                test_predicted = p_sda.predict(x_test_minmax)     
                analysis_scr.append((subset_no, number_of_training, 'l2_reg', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))     
            if settings['DL_S_new_weight_decay'] == 1:
                print 'l1_reg'
                cfg = settings.copy()
                cfg['l1_reg'] =0.001 
                p_sda = Parellel_Sda_factory(cfg)                    
                p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                
                isTest = False #new
                training_predicted = p_sda.predict(x_train_minmax)
                y_train = y_train_minmax                                       
                analysis_scr.append((subset_no, number_of_training, 'l1_reg', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values())) 
                isTest = True #new
                y_test = test_y
                test_predicted = p_sda.predict(x_test_minmax)
                analysis_scr.append((subset_no, number_of_training, 'l1_reg', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))            
                                 
            if settings['DL_S_new_Drop_out'] == 1:
                
                cfg = settings.copy()
                cfg['dropout_factor'] = 0.5
                print 'DL_S_new_Drop_out'
                p_sda = Parellel_Sda_factory(cfg)                    
                p_sda.supervised_training(x_train_minmax, x_validation_minmax, y_train_minmax, y_validation_minmax)
                
                isTest = False #new
                training_predicted = p_sda.predict(x_train_minmax)
                y_train = y_train_minmax                                       
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new_Drop_out', isTest) + tuple(performance_score(train_y_reduced, training_predicted, with_auc_score).values())) 
                isTest = True #new
                y_test = test_y
                test_predicted = p_sda.predict(x_test_minmax)
                analysis_scr.append((subset_no, number_of_training, 'DL_S_new_Drop_out', isTest) + tuple(performance_score(test_y, test_predicted, with_auc_score).values()))            
                         



        report_name = 'DL_handwritten_digits' + '_size_'.join(map(str, hidden_layers_sizes)) + \
                        '_' + str(pretrain_lr) + '_' + str(finetune_lr) + '_' + \
                '_' + str(settings['pretraining_interations']) + '_' + current_date
    saveAsCsv(with_auc_score, report_name, performance_score(test_y, predicted_test_y, with_auc_score), analysis_scr)
    return sda, a_MAE_original, a_MAE_A, a_MAE_B, analysis_scr


# In[9]:

sda, a_MAE_original, a_MAE_A, a_MAE_B, analysis_scr = run_models(cfg)





# In[ ]:




# In[31]:

'''
weights_map_to_input_space = []
StackedNNobject = sda
image_dimension_x = 28*2
image_dimension_y = 28
if isinstance(StackedNNobject, SdA) or isinstance(StackedNNobject, MultipleAEs):
    weights_product = StackedNNobject.dA_layers[0].W.get_value(borrow=True)
    image = PIL.Image.fromarray(tile_raster_images(
        X=weights_product.T,
        img_shape=(image_dimension_x, image_dimension_y), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    sample_image_path = 'hidden_0_layer_weights.png'
    image.save(sample_image_path)
    weights_map_to_input_space.append(weights_product)
    for i_layer in range(1, len(StackedNNobject.dA_layers)):
        i_weigths = StackedNNobject.dA_layers[i_layer].W.get_value(borrow=True)
        weights_product = np.dot(weights_product, i_weigths)
        weights_map_to_input_space.append(weights_product)
        image = PIL.Image.fromarray(tile_raster_images(
        X=weights_product.T,
        img_shape=(image_dimension_x, image_dimension_y), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
        sample_image_path = 'hidden_'+ str(i_layer)+ '_layer_weights.png'
        image.save(sample_image_path)
'''


# In[18]:



