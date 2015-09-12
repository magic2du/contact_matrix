function reportResultsAllVectorsAllNeg_P8_aaIndex_test( ...
                                        logFile,FisherMode, SVMMode, Kernel, Date)

resultsFile = ['resultsAllVectorsAllNeg_P8_aaIndex_test_' ...
                        FisherMode '_' SVMMode '_' Kernel '_' Date '.txt'];
%resultsFile = 'debuggind_09NOV2011.txt';
if exist(resultsFile, 'file')
    command = ['rm -f ' resultsFile];
    system(command);
end
fid = fopen(resultsFile, 'w');


fid2 = fopen(logFile, 'r');
lines = textscan(fid2, '%s', 'delimiter', '\n', 'bufsize', 50*1024*1024);
lines = lines{1};
fclose(fid2);
successedDDIs=[];
errorDDIs=[];
%find those success DDIs
for line_ctr = 1:length(lines)
    logline = lines{line_ctr};
    if length(regexp(logline,'\s','split'))>3
        splitted_words=strsplit(logline, ' ');
        if length(splitted_words)>3
            if  strcmp(splitted_words{3}, 'SUCCESS:')
                ddiname=splitted_words{4};
                ddiname=ddiname(1:end-1);
                successedDDIs{end+1}=ddiname;
            elseif strcmp(splitted_words{3}, 'ERROR:')
                ddiname=splitted_words{4};
                ddiname=ddiname(1:end-1);
                errorDDIs{end+1}=ddiname;
            end
        end
        
    end
end




folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_20NOV2009/dom_dom_ints/'];
%initalize stat
totalPairs=0;
totalauc=0;
totalsvmSens=0;
totalsvmSpec=0;
totalaucBL=0;
totalblSens=0;
totalblSpec=0;
for ddi_ctr = 1:length(successedDDIs)
    
    ddiName = successedDDIs{ddi_ctr};

    ddiPath = [folder3did ddiName '/'];
    folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];

    % load needed data.
    ddiStructFile = [ddiPath 'ddi_str_array.mat'];
    load(ddiStructFile);
    numbPairs = length(ddi_str_array);
    clear ddi_str_array;

    for pairNbr = 1:numbPairs

        try
        resultsFile = ...       
                [folderResults 'AllNeg_P8_aaIndex_test_' ...
                FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.mat'];
            if exist(resultsFile, 'file')
                load(resultsFile);
                totalPairs=totalPairs+1;
                % 11/09/2011 I realized my way of testing predictions results was wrong in
                % that it was forcing sensitivity to be equal to specificity by making the
                % number of predicted positve to be the number of known real positives,
                % which is cheating. I'll change it so that we predict positive whatever
                % lies on the positive side of the hyperplane, negative otherwise.
                [posTestI posTestJ] = ind2sub(size(gtContactMatrix), ...
                                                            find(gtContactMatrix > 0));
                distToHPForRealPos = result(1:length(posTestI));
                distToHPForRealNeg = result((length(posTestI)+1):end);
                svmPredForRealPos = distToHPForRealPos > 0;
                svmPredForRealNeg = distToHPForRealNeg > 0;
                svmTP = length(find(svmPredForRealPos == 1));
                svmFP = length(find(svmPredForRealNeg == 1));
                svmTN = length(find(svmPredForRealNeg == 0));
                svmFN = length(find(svmPredForRealPos == 0));
                svmSens = svmTP/(svmTP+svmFN);
                svmSpec = svmTP/(svmTP+svmFP);
                    if isnan(svmSpec)
                        svmSpec = 0;
                    end

                logline = [ddiName '\t' num2str(pairNbr) ...
                            '\t' num2str(auc, '%0.3f') ...
                            '\t' num2str(svmSens, '%0.3f') ...
                            '\t' num2str(svmSpec, '%0.3f') ...
                            '\t' num2str(aucBL, '%0.3f') ...
                            '\t' num2str(blSens, '%0.3f') ...
                            '\t' num2str(blSpec, '%0.3f') '\n'];
                totalauc=auc+totalauc;
                totalsvmSens=totalsvmSens+svmSens;
                totalsvmSpec=totalsvmSpec+svmSpec;
                totalaucBL=totalaucBL+aucBL;
                totalblSens=totalblSens+blSens;
                totalblSpec=totalblSpec+blSpec;

                fprintf(fid, logline);
            end

        catch exc
            c = clock;
            time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                    num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                    num2str(c(5), '%0.0d')];
            logline = [time ' ERROR: ' ddiName ...
                                            '. Message: ' exc.message '\n'];
            fprintf(logline);
        end

    end

end
fprintf('The total number of DDIs is: %d\n', length(successedDDIs)+length(errorDDIs));
fprintf('The total number of successed DDIs is: %d\n', size(successedDDIs,2));
fprintf('The number of training sequence pairs LOO is: %d\n', totalPairs);
fprintf('average AUC is: %f\n', totalauc/totalPairs);
fprintf('average Recall is: %f\n', totalsvmSens/totalPairs);
fprintf('average Precision is: %f\n', totalsvmSpec/totalPairs);
fprintf('average baseline AUC is: %f\n', totalaucBL/totalPairs);
fprintf('average baseline Recall is: %f\n', totalblSens/totalPairs);
fprintf('average baseline Precision is: %f\n', totalblSpec/totalPairs);


fclose(fid);
                    
return;
