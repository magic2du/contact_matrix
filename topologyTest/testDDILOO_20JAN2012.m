function [meanAUC meanZscore meanPvalue] = ...
                    testDDILOO_20JAN2012(ddi_folder, FisherMode, SVMMode)

%ddi_folder = '/home/alvaro/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/Homoserine_dh_int_NAD_binding_3';

% load needed data.
ddiStructFile = [ddi_folder '/ddi_str_array.mat'];
load(ddiStructFile);
numbPairs = length(ddi_str_array);
clear ddi_str_array;

AUC_Zscore_Pvalue_Array = [];
startTime = clock;
for pairCtr = 1:numbPairs
    
    % if this is taking too long, go on! (limit is 2 hours).
    eTime = etime(clock, startTime);
    if eTime > 60*2120
        error('This DDI was taking too long.');
    end
    
    [auc Zscore Pvalue] = ...
        testSeqPairLOO_20JAN2012(ddi_folder, pairCtr, FisherMode, SVMMode);
    AUC_Zscore_Pvalue_Array(end+1, :) = [auc Zscore Pvalue];
end
meanAUC = mean(AUC_Zscore_Pvalue_Array(:, 1));
meanZscore = mean(AUC_Zscore_Pvalue_Array(:, 2));
meanPvalue = mean(AUC_Zscore_Pvalue_Array(:, 3));

% print results to file.
summaryFile = [ddi_folder '/' FisherMode SVMMode '.summary'];
save(summaryFile, 'AUC_Zscore_Pvalue_Array', '-ascii');

return;
