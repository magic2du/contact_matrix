function build_matrix_model_with_contactMaps(ddiName, FisherMode, SVMMode, Kernel)
%function auc = runSeqPairAllVectorsChooseNegRand_02NOV2011()
% pairNbr is the current sequence number doing the LOO cross validation.
% SVM mode 'SVMLIGHT', 

folder3did = ['/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/'];
ddiPath= [folder3did ddiName '/'];
folderResults = ddiPath;



% get contact matrix baseline pred. from the training seq. pairs. Baseline
% prediction will come from the family, except the pair being tested.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
contactMatrixPath = ...
        [folder3did  ddiName '/contactMapsBag.mat'];
load(contactMatrixPath); % contactMapsBag.
contactMapsBag(end) = []; % the last one is the average.


% Fisher vectors. For all the training sequence pairs, go to the
% corresponding contact matrix, and get from there positive and negative
% Fisher vector pairs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load([ddiPath 'FisherA.mat']); % AFisherM0[1]Array, AconstFisherM0[1]Array.
load([ddiPath 'FisherB.mat']); % BFisherM0[1]Array, BconstFisherM0[1]Array.

load([ddiPath 'ddi_str_array.mat']); % AFisherM0[1]Array, AconstFisherM0[1]Array.

% read HMM structures for the two domains involved in the interaction.
domA = ddi_str_array{1}.domainA;
domB = ddi_str_array{1}.domainB;
hmmA = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/' ...
                        'PFAM_2008/SINGLE_FILES/' domA '.pfam']);
hmmB = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/' ...
                        'PFAM_2008/SINGLE_FILES/' domB '.pfam']);

FamilyPO = [];
FamilyNE = [];
trainPO = [];
trainNE = [];

%for seqCtr = setdiff(1:length(contactMapsBag), pairNbr)
% use all the examples as training examples.
for idxCtr = 1:length(ddi_str_array)
    seqCtr = idxCtr;
    
    %compute the HMM aligned sequence ignoring insert, should be HMM length
    seqA = ddi_str_array{seqCtr}.ASequence;
    seqB = ddi_str_array{seqCtr}.BSequence;
    % seqA
    [scoreA, algnA] = hmmprofalign(hmmA, seqA, 'flanks', true);
    indDelA = strfind(algnA, '-');
    indMatchDelA = union(indDelA, regexp(algnA, '[A-Z]'));
    conservedProteinSequenceA = algnA(indMatchDelA); 
    ddi_str_array{seqCtr}.conservedProteinSequenceA=conservedProteinSequenceA;
    % seqB
    [scoreB, algnB] = hmmprofalign(hmmB, seqB, 'flanks', true);
    indDelB = strfind(algnB, '-');
    indMatchDelB = union(indDelB, regexp(algnB, '[A-Z]'));
    conservedProteinSequenceB = algnB(indMatchDelB); 
    ddi_str_array{seqCtr}.conservedProteinSequenceB=conservedProteinSequenceB;
    
    %get FisherVectors
    command = ['AFisherVectors = A' FisherMode 'Array{seqCtr};'];
    eval(command);
    
    command = ['BFisherVectors = B' FisherMode 'Array{seqCtr};'];
    eval(command);
    
    % indexes of positive and negative examples.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [posI posJ] = ind2sub(size(contactMapsBag{seqCtr}), ...
                            find(contactMapsBag{seqCtr} > 0));
    [negI negJ] = ind2sub(size(contactMapsBag{seqCtr}), ...
                            find(contactMapsBag{seqCtr} == 0));
    
    % create positive and negative training sets.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PO=[];
    NE=[];
    % create positive and negative training sets.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    PO = [AFisherVectors(posI, :) BFisherVectors(posJ, :)];
    NE = [AFisherVectors(negI, :) BFisherVectors(negJ, :)];
    FamilyPO = [FamilyPO; PO];
%    FamilyNE = [FamilyNE; NE];   we don't use this in this problem.

    % chose negatives as in the the ratio default is 1.

    numbNegTrain = size(PO, 1) * 1;
    r=randperm(size(NE, 1));
    r=r(1:numbNegTrain);
    indTrainNE=sort(r);
    selectedNE = NE(indTrainNE, :);
    trainNE=[trainNE; selectedNE];
end
trainPO=FamilyPO; %use all the Positive examples as 


% Train SVM.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%numbNegTrain = size(FamilyPO, 1);
% question: how many negatives? clustering?

%r=randperm(size(FamilyNE, 1));
%r=r(1:numbNegTrain);
%indTrainNE=sort(r);
%FamilyNE = FamilyNE(indTrainNE, :);
length(trainPO)
length(trainNE)
if strcmp(SVMMode, 'SVMLIGHT')
% SVMLIGHT.
svmlightFolder = ...
'/home/du/Protein_Protein_Interaction_Project/svm_light_linux64_2013/';

trainFile = ...
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' 'build_matrix_model_with_contactMaps.train']
modelFile = ...
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_' 'build_matrix_model_with_contactMaps.model'];
% write_SVM_file(FamilyPO, FamilyNE, trainFile);
write_SVM_file(trainPO, trainNE, trainFile);

    if strcmp(Kernel, 'RBF')
	    command = ...
            [svmlightFolder 'svm_learn -t 2 -g 1 ' trainFile ' ' modelFile];
    elseif strcmp(Kernel, 'POLY')
	    command = ...
            [svmlightFolder 'svm_learn -t 1 -d 3 ' trainFile ' ' modelFile];
    elseif strcmp(Kernel, 'LINEAR')
	    command = ...
            [svmlightFolder 'svm_learn -t 0 ' trainFile ' ' modelFile];
    else
	    error('ERROR');
    end
    system(command);
'model file built success'
else
% MATLAB.
%Training = [FamilyPO; FamilyNE];
Training = [trainPO; trainNE];

    Group = [ones(size(trainPO, 1), 1); zeros(size(trainNE, 1), 1)];
    if strcmp(Kernel, 'RBF')
	    SVMStruct = svmtrain(Training, Group, 'kernel_function', 'rbf');
    elseif strcmp(Kernel, 'POLY')
	    SVMStruct = svmtrain(Training, Group, 'kernel_function', 'polynomial');
    else
	    SVMStruct = svmtrain(Training, Group, 'kernel_function', 'linear');
    end

end

return;
