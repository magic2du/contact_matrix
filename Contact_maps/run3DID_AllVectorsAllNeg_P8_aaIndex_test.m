function run3DID_AllVectorsAllNeg_P8_aaIndex_test( ...
                                            file_ddis, FisherMode, SVMMode, Kernel)

diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_run3DID_AllVectorsAllNeg_' ...
            FisherMode '_' SVMMode '_' Kernel '_' dateFormatted '.txt'];
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

% we'll only do the ddis in this dataset.
%file_ddis = 'dataset_25JAN2011.txt';
%file_ddis = 'dataset_09AUG2011_FinContMatrIphmmFisher.txt';
%file_ddis = 'ddisFinishedByAll_11AUG2011.txt';
%file_ddis = 'prototypeDDI.txt';
%file_ddis = 'finishedDDIs_GraphClustering_18OCT2011.txt';
%file_ddis = 'finishedDDIs_GraphClustering_31OCT2011.txt';
%file_ddis = 'goodClusterings_02NOV2011.txt';
%file_ddis = ...
%'finishedDDIs_AllVectorsChooseNegRand_FisherM1_SVMLIGHT_POLY_02NOV2011.txt';
fid = fopen(file_ddis, 'r');
cell_ddis = textscan(fid, '%s', 'delimiter', '\n');
cell_ddis = cell_ddis{1};
fclose(fid);

currDir = pwd;
successCtr = 0;
matlabpool open local 8
parfor ddi_ctr = 1:length(cell_ddis)
    cd(currDir);
    %{
    ddi_line = textscan(cell_ddis{ddi_ctr}, '%s %s');
    domAPfam = ddi_line{1}{1};
    domBPfam = ddi_line{2}{1};
    ddiName = [domAPfam '_int_' domBPfam];
    %}
    ddiName = cell_ddis{ddi_ctr};
    
    try
        
        %meanAUC = runDDIAllVectorsChooseNegRand_02NOV2011( ...
        meanAUC = runDDIAllVectorsAllNeg_P8_aaIndex_test( ...
                                ddiName, FisherMode, SVMMode, Kernel);
        
        % finished! report SUCCESS to log file.
        successCtr = successCtr + 1;
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = ['\n\n\n' time ' SUCCESS: ' ddiName ...
                    ', meanAUC = ' num2str(meanAUC, '%0.3f') '.\n\n\n'];
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
matlabpool close
cd(currDir);
c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' FINISHED.\n'];
fprintf(logline);

diary off;

return;
