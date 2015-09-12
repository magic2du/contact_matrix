function     [aucBL_Array, blSens_Array, blSpec_Array, ...
                    auc_Array, svmSens_Array, svmSpec_Array, test_f_measure_Array,...
                    trainAccuracy_Array, trainRecall_Array, trainPrecision_Array, train_f_measure_Array]  = runDDI_remote_ChooseNegRand_aaIndex_test_load_DL_cross ...
                                (ddiName, trainPO, trainNE, POTest, NETest, FisherMode, SVMMode, Kernel, choseNegRatio, choseNegRatioTest)



aucBL_Array = [];
blSens_Array = [];
blSpec_Array = [];
auc_Array = [];
svmSens_Array = [];
svmSpec_Array = [];
test_f_measure_Array = [];
trainAccuracy_Array = [];
trainRecall_Array = [];
trainPrecision_Array = [];
train_f_measure_Array = [];

trainGroundTruth = [ones(size(trainPO, 1), 1); ...
                -1*ones(size(trainNE, 1), 1)];
testGroundTruth = [ones(size(POTest, 1), 1); ...
                -1*ones(size(NETest, 1), 1)];

%train and testing
if strcmp(SVMMode, 'SVMLIGHT')
    % SVMLIGHT.
    % test on training set.
    % SVMLIGHT.
    svmlightFolder = ...
    '/home/du/Protein_Protein_Interaction_Project/3did_20NOV2009/svm_light/';
    folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
    trainFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.train'];
    modelFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.model'];
    % write_SVM_file(FamilyPO, FamilyNE, trainFile);
    write_SVM_file(trainPO, trainNE, trainFile);
    if strcmp(Kernel, 'RBF')
        command = ...
            [svmlightFolder 'svm_learn -t 2 -g 1 -c ' num2str(C) ' -j ' num2str(Jj) ' '  trainFile ' ' modelFile];
    elseif strcmp(Kernel, 'POLY')
        command = ...
            [svmlightFolder 'svm_learn -t 1 -d 3 -c ' num2str(C) ' -j ' num2str(Jj) ' '  trainFile ' ' modelFile];
    elseif strcmp(Kernel, 'LINEAR')
        command = ...
            [svmlightFolder 'svm_learn -t 0 ' trainFile ' ' modelFile];
    else
        error('ERROR');
    end
    system(command);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % SVMLIGHT.
    % test on training set.
    testFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.testOnTrain'];
    resultFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.resultOnTrain'];
    write_SVM_file(trainPO, trainNE, testFile);
    command = ...
    [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
    system(command);
    trainResult = load(resultFile);
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
    % for training performance.
    trainResult0_1=zeros(length(trainResult), 1);
    trainResult0_1(find(trainResult>=0))=1;
    trainGroundTruth(find(trainGroundTruth==-1))=0; % change the format to 0 and 1;
    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(trainGroundTruth, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)
     
    if isnan(precision)
        precision = 0;
    end
    if isnan(f_measure)
        f_measure = 0;
    end
    fprintf('Training AUC: %0.3f%%\n', trainAUC * 100);
    fprintf('Training accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Training recall: %0.3f%%\n', recall * 100);
    fprintf('Training precision: %0.3f%%\n', precision * 100);
    fprintf('Training f_measure: %0.3f%%\n', f_measure * 100);
    trainAccuracy=accuracy;
    trainRecall=recall;
    trainPrecision=precision;
    train_f_measure=f_measure;

    % test on test set:
    testFile = ...
        [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.test'];

    resultFile = [folderResults FisherMode '_'  SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' num2str(choseNegRatioTest) '_remote_ChooseNegRand_aaIndex_test_load_DL_cross.result'];

    write_SVM_file(POTest, NETest, testFile);
    command = ...
    [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
    system(command);
    testResult = load(resultFile);
    auc = roc(testResult, testGroundTruth, 'nofigure');
    % for testing set performance.
    testResult0_1=zeros(length(testResult),1);
    testResult0_1(find(testResult>=0))=1;
    testGroundTruth(find(testGroundTruth==-1))=0; % change the format to 0 and 1;
    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testGroundTruth, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
    if isnan(precision)
        precision = 0;
    end
    if isnan(f_measure)
        f_measure = 0;
    end
    fprintf('Testing AUC: %0.3f%%\n', auc * 100);
    fprintf('Testing accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Testing recall: %0.3f%%\n', recall * 100);
    fprintf('Testing precision: %0.3f%%\n', precision * 100);
    fprintf('Testing f_measure: %0.3f%%\n', f_measure * 100);
    svmSens=recall;
    svmSpec=precision;
    test_f_measure=f_measure;    
    
   
elseif strcmp(SVMMode, 'DL')



    trainData=[trainPO; trainNE];
    trainLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
    testData=[POTest; NETest];
    testLabel=[ones(size(POTest, 1), 1); ...
				    zeros(size(NETest, 1), 1)];
    % run the DL algorithms
    [trainResult, testResult]= SAE_remote_ChooseNegRand_aaIndex_test_load_DL_cross(trainData, trainLabel, testData, testLabel);
        
    auc = roc(testResult, testGroundTruth, 'nofigure');
    trainAUC = roc(trainResult, trainGroundTruth, 'nofigure');
    
    % for training performance.
    trainResult0_1=zeros(length(trainResult), 1);
    trainResult0_1(find(trainResult>=0))=1;
    trainGroundTruth(find(trainGroundTruth==-1))=0; % change the format to 0 and 1;
    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(trainGroundTruth, trainResult0_1); %Evaluate(ACTUAL,PREDICTED)
     
    if isnan(precision)
        precision = 0;
    end
    if isnan(f_measure)
        f_measure = 0;
    end
    fprintf('Training AUC: %0.3f%%\n', trainAUC * 100);
    fprintf('Training accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Training recall: %0.3f%%\n', recall * 100);
    fprintf('Training precision: %0.3f%%\n', precision * 100);
    fprintf('Training f_measure: %0.3f%%\n', f_measure * 100);
    trainAccuracy=accuracy;
    trainRecall=recall;
    trainPrecision=precision;
    train_f_measure=f_measure;
    % for testing set performance.
    testResult0_1=zeros(length(testResult),1);
    testResult0_1(find(testResult>=0))=1;
    testGroundTruth(find(testGroundTruth==-1))=0; % change the format to 0 and 1;
    [accuracy, sensitivity, specificity, precision, recall, f_measure, gmean]=Evaluate(testGroundTruth, testResult0_1); %Evaluate(ACTUAL,PREDICTED)
    if isnan(precision)
        precision = 0;
    end
    if isnan(f_measure)
        f_measure = 0;
    end
    fprintf('Testing AUC: %0.3f%%\n', auc * 100);
    fprintf('Testing accuracy: %0.3f%%\n', accuracy * 100);
    fprintf('Testing recall: %0.3f%%\n', recall * 100);
    fprintf('Testing precision: %0.3f%%\n', precision * 100);
    fprintf('Testing f_measure: %0.3f%%\n', f_measure * 100);
    svmSens=recall;
    svmSpec=precision;
    test_f_measure=f_measure;    
    
    
else 
    error('wrong MODE');
end
blSens = 0;
blSpec = 0;
aucBL = 0;

% collecting result
aucBL_Array(end+1, :) = aucBL;
blSens_Array(end+1, :) = blSens;
blSpec_Array(end+1, :) = blSpec;
auc_Array (end+1, :)= auc;
svmSens_Array(end+1, :) = svmSens;
svmSpec_Array(end+1, :) = svmSpec;
test_f_measure_Array(end+1, :) = test_f_measure;
trainAccuracy_Array(end+1, :) = trainAccuracy;
trainRecall_Array(end+1, :) = trainRecall;
trainPrecision_Array(end+1, :) = trainPrecision;
train_f_measure_Array(end+1, :) = train_f_measure;


%fprintf(['\n\n\nFinished ' ddiName ', ...
 %                       meanAUC = ' num2str(meanAUC, '%0.3f') '.\n\n\n']);

% print results to file.
%summaryFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
%                            FisherMode '_' SVMMode '_' Kernel '.summary'];
%save(summaryFile, 'AUC_Array', '-ascii');

return;
