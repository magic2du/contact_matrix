function [trainResult, testResult]=runSVMLight(newTrainX,train_y, newTestX, test_y, ddiPath)
% train_y and testLable
    Kernel='LINEAR';
    svmlightFolder = '/home/michael/libs/svmlight/';
    folderResults = ddiPath;
    trainFile = ...
        [folderResults 'tmp.train'];
    modelFile = ...
        [folderResults 'tmp.model'];
    % Training the model;
    %write_SVM_file_from_X_plus_Y(data_X_plus_Y, trainFile);
    m= size(newTrainX, 1);
    mTest=size(newTestX, 1);
    trainGroundTruth = train_y;
    testGroundTruth = test_y;

    write_SVM_file(newTrainX(1:m/2, :), newTrainX(m/2+1:end , :), trainFile);
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
    resultFile = [folderResults 'tmp.result'];
    command = ...
    [svmlightFolder 'svm_classify ' trainFile ' ' modelFile ' ' resultFile];
    system(command);
    trainResult = load(resultFile);

    % test on test set:
    testFile = ...
        [folderResults 'tmp.test'];

    resultFile = [folderResults 'tmp.result'];

    write_SVM_file(newTestX(1:mTest/2, :), newTestX(mTest/2+1:end , :), testFile);
    command = ...
    [svmlightFolder 'svm_classify ' testFile ' ' modelFile ' ' resultFile];
    system(command);
    testResult = load(resultFile);
end
