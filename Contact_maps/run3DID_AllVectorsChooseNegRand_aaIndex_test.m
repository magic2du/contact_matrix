function run3DID_AllVectorsChooseNegRand_aaIndex_test( ...
                                            file_ddis, FisherMode, SVMMode, Kernel, choseNegRatio)

diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_run3DID_AllVectorsChooseNegRand_aaIndex_test_' ...
            FisherMode '_' SVMMode '_' Kernel '_' num2str(choseNegRatio) '_' dateFormatted '.txt'];
if exist(logfile_name, 'file')
    command = ['rm -f ' logfile_name];
    system(command);
end
diary(logfile_name);

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' STARTED.\n'];
fprintf(logline);
%'finishedDDIs_AllVectorsChooseNegRand_FisherM1_SVMLIGHT_POLY_02NOV2011.txt';
fid = fopen(file_ddis, 'r');
cell_ddis = textscan(fid, '%s', 'delimiter', '\n');
cell_ddis = cell_ddis{1};
fclose(fid);
currDir = pwd;

%initialize parameters
successCtr = 0;
aucBL_Array_W=[];
blSens_Array_W=[];
blSpec_Array_W=[];
auc_Array_W=[];
svmSens_Array_W=[];
svmSpec_Array_W=[];
test_f_measure_Array_W= [];
trainAccuracy_Array_W=[];
trainRecall_Array_W=[];
trainPrecision_Array_W=[];
train_f_measure_Array_W=[]; 


%matlabpool open local 5 %number of parallel machine max=8
for ddi_ctr = 1:length(cell_ddis)
    ddiName = cell_ddis{ddi_ctr};
    
    try
        
        [aucBL_Array, blSens_Array, blSpec_Array, ...
                    auc_Array, svmSens_Array, svmSpec_Array, test_f_measure_Array,...
                    trainAccuracy_Array, trainRecall_Array, trainPrecision_Array, train_f_measure_Array]  = runDDIAllVectorsChooseNegRand_aaIndex_test ...
                                (ddiName, FisherMode, SVMMode, Kernel, choseNegRatio);
        aucBL_Array_W=[aucBL_Array_W; aucBL_Array];
        blSens_Array_W=[blSens_Array_W; blSens_Array];
        blSpec_Array_W=[ blSpec_Array_W; blSpec_Array];
        auc_Array_W=[auc_Array_W; auc_Array];
        svmSens_Array_W=[svmSens_Array_W; svmSens_Array];
        svmSpec_Array_W=[svmSpec_Array_W; svmSpec_Array];
        test_f_measure_Array_W= [test_f_measure_Array_W; test_f_measure_Array];
        trainAccuracy_Array_W=[trainAccuracy_Array_W; trainAccuracy_Array];
        trainRecall_Array_W=[trainRecall_Array_W; trainRecall_Array];
        trainPrecision_Array_W=[trainPrecision_Array_W; trainPrecision_Array];
        train_f_measure_Array_W=[train_f_measure_Array_W; train_f_measure_Array];
        
        % finished! report SUCCESS to log file.

        successCtr = successCtr + 1;
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = ['\n\n\n' time ' SUCCESS: ' ddiName ...
                    '.\n\n\n'];
        fprintf(logline);
        
        %{
        if successCtr == 20
            % force end.
            break;
        end
        %}
        
    catch exc
        cd(currDir);
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = [time ' ERROR: ' ddiName ...
                                        '. Message: ' exc.message '\n'];
        fprintf(logline);
        % print the stack.
        for stackCtr = length(exc.stack):-1:1
            fprintf([exc.stack(stackCtr).file '; ' ...
                        exc.stack(stackCtr).name '; ' ...
                        num2str(exc.stack(stackCtr).line) '\n']);
        end
    end
    
end
%matlabpool close
cd(currDir);
c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' FINISHED.\n'];
fprintf(logline);
% report the final result of LOO, includes baseline, testing, training error. 
fprintf('Number of total DDI is: %d\n',length(cell_ddis));
fprintf('Number of successed DDI is: %d\n',successCtr);
fprintf('Number of total sequence is: %d\n',length(aucBL_Array_W));

fprintf('The baseline AUC: %0.3f%%\n', mean(aucBL_Array_W) * 100);
fprintf('The baseline recall: %0.3f%%\n', mean(blSens_Array_W) * 100);
fprintf('The baseline precision: %0.3f%%\n', mean(blSpec_Array_W) * 100);

fprintf('Testing AUC: %0.3f%%\n',  mean(auc_Array_W) * 100);
fprintf('Testing recall: %0.3f%%\n', mean(svmSens_Array_W) * 100);
fprintf('Testing precision: %0.3f%%\n', mean(svmSpec_Array_W) * 100);
fprintf('Testing f_measure: %0.3f%%\n', mean(test_f_measure_Array_W) * 100);

fprintf('Training accuracy: %0.3f%%\n', mean(trainAccuracy_Array_W) * 100);
fprintf('Training recall: %0.3f%%\n', mean(trainRecall_Array_W) * 100);
fprintf('Training precision: %0.3f%%\n',  mean(trainPrecision_Array_W)* 100);
fprintf('Training f_measure: %0.3f%%\n', mean(train_f_measure_Array_W) * 100);

diary off;

return;
