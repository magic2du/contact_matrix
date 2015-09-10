function [meanAUC meanZscore meanPvalue] = ...
                    testDDILOO_topology_level_parallel(ddi_folder, FisherMode, SVMMode)

%ddi_folder = '/home/alvaro/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/Homoserine_dh_int_NAD_binding_3';
TopologyListFile=[ddi_folder '/topologyList.txt'];
fid = fopen(TopologyListFile, 'r');
topologyList = textscan(fid, '%s', 'delimiter', '\n');
topologyList = topologyList{1};
fclose(fid);
%loop through topology level
for topology = 1:length(topologyList)
	topologyName=topologyList{topology};
	% load needed data.
	ddiStructFile = [ddi_folder '/' topologyName '/ddi_str_array.mat'];
	load(ddiStructFile);
	numbPairs = length(ddi_str_array);
	clear ddi_str_array;

	AUC_Zscore_Pvalue_Array = [];
	startTime = clock;
	for pairCtr = 1:numbPairs
	    
	    % if this is taking too long, go on! (limit is 2 hours).
	    eTime = etime(clock, startTime);
	    if eTime > 60*60*72
		error('This DDI was taking too long.');
	    end
	    
	    [auc Zscore Pvalue] = ...
		testSeqPairLOO_topology_level_parallel_remote([ddi_folder '/' topologyName], pairCtr, FisherMode, SVMMode);
	    AUC_Zscore_Pvalue_Array(end+1, :) = [auc Zscore Pvalue];
	end
	meanAUC = mean(AUC_Zscore_Pvalue_Array(:, 1));
	meanZscore = mean(AUC_Zscore_Pvalue_Array(:, 2));
	meanPvalue = mean(AUC_Zscore_Pvalue_Array(:, 3));

	% print results to file.
	summaryFile = [ddi_folder '/' topologyName '/' FisherMode SVMMode '.summary'];
	save(summaryFile, 'AUC_Zscore_Pvalue_Array', '-ascii');
end
return;