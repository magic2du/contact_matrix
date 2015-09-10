function ddiNegativeSequencesAndFisher_topology_level_remote(ddi_folder)
%topologyList = The topolgy list in the topolgyListFile;

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
	iphmmFile = [ddi_folder '/' topologyName '/iphmmA.mat'];
	load(iphmmFile);
	iphmmFile = [ddi_folder '/' topologyName '/iphmmB.mat'];
	load(iphmmFile);
	ConsSeqFile = [ddi_folder '/' topologyName '/ConsSeqA.mat'];
	load(ConsSeqFile);
	ConsSeqFile = [ddi_folder '/' topologyName '/ConsSeqB.mat'];
	load(ConsSeqFile);
	intSitesFile = [ddi_folder '/' topologyName '/int_sitesA.mat'];
	load(intSitesFile);
	intSitesFile = [ddi_folder '/' topologyName '/int_sitesB.mat'];
	load(intSitesFile);

	currDir = pwd;
	cd /home/du/Protein_Protein_Interaction_Project/ipHMMs;

	% train sequences.
	numbTrain = length(ddi_str_array);

	negTrainA = {};
	negTrainB = {};
	AFisherM0ArrayNegTrain = {};
	AFisherM1ArrayNegTrain = {};
	AconstFisherM0ArrayNegTrain = {};
	AconstFisherM1ArrayNegTrain = {};
	BFisherM0ArrayNegTrain = {};
	BFisherM1ArrayNegTrain = {};
	BconstFisherM0ArrayNegTrain = {};
	BconstFisherM1ArrayNegTrain = {};
	negTrainCtr = 0;
	while negTrainCtr < numbTrain
	    try
	    % domA.
	    rand_aa = int2aa(randi(20, size(int_sitesA)));
	    randSeqA = ConsSeqA;
	    randSeqA(int_sitesA) = rand_aa;
	    [AFisherM0 AFisherM1 AconstFisherM0 AconstFisherM1] = ...
		                        calculateFisherVector(randSeqA, iphmmA);
	    
	    % domB.
	    rand_aa = int2aa(randi(20, size(int_sitesB)));
	    randSeqB = ConsSeqB;
	    randSeqB(int_sitesB) = rand_aa;
	    [BFisherM0 BFisherM1 BconstFisherM0 BconstFisherM1] = ...
		                        calculateFisherVector(randSeqB, iphmmB);
	    
	    negTrainA(end+1) = {randSeqA};
	    negTrainB(end+1) = {randSeqB};
	    AFisherM0ArrayNegTrain(end+1) = {AFisherM0};
	    AFisherM1ArrayNegTrain(end+1) = {AFisherM1};
	    AconstFisherM0ArrayNegTrain(end+1) = {AconstFisherM0};
	    AconstFisherM1ArrayNegTrain(end+1) = {AconstFisherM1};
	    BFisherM0ArrayNegTrain(end+1) = {BFisherM0};
	    BFisherM1ArrayNegTrain(end+1) = {BFisherM1};
	    BconstFisherM0ArrayNegTrain(end+1) = {BconstFisherM0};
	    BconstFisherM1ArrayNegTrain(end+1) = {BconstFisherM1};
	    
	    negTrainCtr = negTrainCtr + 1;
	    
	    catch
		% being random sequences, the alignment to the iphmm sometimes
		% doesn't work, so we'll just disregard that sequence and try a new
		% one.
		continue;
	    end
	end

	% test sequences.
	numbTest = 100;

	negTestA = {};
	negTestB = {};
	AFisherM0ArrayNegTest = {};
	AFisherM1ArrayNegTest = {};
	AconstFisherM0ArrayNegTest = {};
	AconstFisherM1ArrayNegTest = {};
	BFisherM0ArrayNegTest = {};
	BFisherM1ArrayNegTest = {};
	BconstFisherM0ArrayNegTest = {};
	BconstFisherM1ArrayNegTest = {};
	negTestCtr = 0;
	while negTestCtr < numbTest
	    try
	    % domA.
	    rand_aa = int2aa(randi(20, size(int_sitesA)));
	    randSeqA = ConsSeqA;
	    randSeqA(int_sitesA) = rand_aa;
	    [AFisherM0 AFisherM1 AconstFisherM0 AconstFisherM1] = ...
		                        calculateFisherVector(randSeqA, iphmmA);
	    
	    % domB.
	    rand_aa = int2aa(randi(20, size(int_sitesB)));
	    randSeqB = ConsSeqB;
	    randSeqB(int_sitesB) = rand_aa;
	    [BFisherM0 BFisherM1 BconstFisherM0 BconstFisherM1] = ...
		                        calculateFisherVector(randSeqB, iphmmB);
	    
	    negTestA(end+1) = {randSeqA};
	    negTestB(end+1) = {randSeqB};
	    AFisherM0ArrayNegTest(end+1) = {AFisherM0};
	    AFisherM1ArrayNegTest(end+1) = {AFisherM1};
	    AconstFisherM0ArrayNegTest(end+1) = {AconstFisherM0};
	    AconstFisherM1ArrayNegTest(end+1) = {AconstFisherM1};
	    BFisherM0ArrayNegTest(end+1) = {BFisherM0};
	    BFisherM1ArrayNegTest(end+1) = {BFisherM1};
	    BconstFisherM0ArrayNegTest(end+1) = {BconstFisherM0};
	    BconstFisherM1ArrayNegTest(end+1) = {BconstFisherM1};
	    
	    negTestCtr = negTestCtr + 1;
	    
	    catch
		% being random sequences, the alignment to the iphmm sometimes
		% doesn't work, so we'll just disregard that sequence and try a new
		% one.
		continue;
	    end
	end

	cd(currDir);

	saveFile = [ddi_folder '/' topologyName '/negTrainA.mat'];
	save(saveFile, 'negTrainA');
	saveFile = [ddi_folder '/' topologyName '/negTrainB.mat'];
	save(saveFile, 'negTrainB');
	saveFile = [ddi_folder '/' topologyName '/negTestA.mat'];
	save(saveFile, 'negTestA');
	saveFile = [ddi_folder '/' topologyName '/negTestB.mat'];
	save(saveFile, 'negTestB');
	saveFile = [ddi_folder '/' topologyName '/FisherANegTrain.mat'];
	save(saveFile, 'AFisherM0ArrayNegTrain', 'AFisherM1ArrayNegTrain', ...
		    'AconstFisherM0ArrayNegTrain', 'AconstFisherM1ArrayNegTrain');
	saveFile = [ddi_folder '/' topologyName '/FisherBNegTrain.mat'];
	save(saveFile, 'BFisherM0ArrayNegTrain', 'BFisherM1ArrayNegTrain', ...
		    'BconstFisherM0ArrayNegTrain', 'BconstFisherM1ArrayNegTrain');
	saveFile = [ddi_folder '/' topologyName '/FisherANegTest.mat'];
	save(saveFile, 'AFisherM0ArrayNegTest', 'AFisherM1ArrayNegTest', ...
		    'AconstFisherM0ArrayNegTest', 'AconstFisherM1ArrayNegTest');
	saveFile = [ddi_folder '/' topologyName '/FisherBNegTest.mat'];
	save(saveFile, 'BFisherM0ArrayNegTest', 'BFisherM1ArrayNegTest', ...
		    'BconstFisherM0ArrayNegTest', 'BconstFisherM1ArrayNegTest');
end
return;