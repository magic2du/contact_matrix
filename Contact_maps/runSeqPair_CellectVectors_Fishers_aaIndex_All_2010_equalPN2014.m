function [V, numPos, numNeg] = runSeqPair_CellectVectors_Fishers_aaIndex_All_2010_equalPN2014 ...
                        (ddiName, pairNbr)
%function auc = runSeqPairAllVectorsChooseNegRand_02NOV2011()

folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_15OCT2010/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
folderResults = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/dom_dom_ints/' ddiName '/'];
if ~exist(folderResults, 'dir')
	mkdir(folderResults);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
contactMatrixPath = ...
        ['/home/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/ContactMapExamples/' ...
                                        ddiName '/contactMapsBag.mat'];
load(contactMatrixPath); % contactMapsBag.
contactMapsBag(end) = []; % the last one is the average.
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

% Testing.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gtContactMatrix = contactMapsBag{pairNbr}; %ground truth.
command = ['AFisherM0Vectors = A' 'FisherM0' 'Array{pairNbr};'];
eval(command);
command = ['BFisherM0Vectors = B' 'FisherM0' 'Array{pairNbr};'];
eval(command);
command = ['AFisherM1Vectors = A' 'FisherM1' 'Array{pairNbr};'];
eval(command);
command = ['BFisherM1Vectors = B' 'FisherM1' 'Array{pairNbr};'];
eval(command);

% create positive and negative test sets.
[posTestI posTestJ] = ind2sub(size(gtContactMatrix), ...
                                            find(gtContactMatrix > 0));
%[negTestI negTestJ] = ind2sub(size(gtContactMatrix), ...
%                                            find(gtContactMatrix == 0));
%

[neg_mesh_row, neg_mesh_col]  = meshgrid(posTestI, posTestJ);
coord_p = [posTestI, posTestJ];
coord_n = [neg_mesh_row(:), neg_mesh_col(:)];
[tf, loc] = ismember(coord_n, coord_p, 'rows');
dup = find(tf==1);
coord_n(dup, :) = [];
negTestI = coord_n(:, 1);
negTestJ = coord_n(:, 2);

if length(negTestI) == 0
	[contactMatrixRows, contactMatrixCols] = size(gtContactMatrix);
	negTestJ = posTestJ(1);
	negTestI = posTestI(1);
	if length(posTestI) == 1
		while(gtContactMatrix(negTestI, negTestJ) > 0)
			negTestJ = floor((contactMatrixCols - 1) * rand() + 1);
        	end
	else
		while(gtContactMatrix(negTestI, negTestJ) > 0)
			negTestI = floor((contactMatrixRows - 1) * rand() + 1);
        	end
	end

	
	
end

%compute the HMM aligned sequence ignoring insert, should be HMM length
seqA = ddi_str_array{pairNbr}.ASequence;
seqB = ddi_str_array{pairNbr}.BSequence;
% seqA
[scoreA, algnA] = hmmprofalign(hmmA, seqA, 'flanks', true);
indDelA = strfind(algnA, '-');
indMatchDelA = union(indDelA, regexp(algnA, '[A-Z]'));
conservedProteinSequenceA = algnA(indMatchDelA); 
ddi_str_array{pairNbr}.conservedProteinSequenceA=conservedProteinSequenceA;
% seqB
[scoreB, algnB] = hmmprofalign(hmmB, seqB, 'flanks', true);
indDelB = strfind(algnB, '-');
indMatchDelB = union(indDelB, regexp(algnB, '[A-Z]'));
conservedProteinSequenceB = algnB(indMatchDelB); 
ddi_str_array{pairNbr}.conservedProteinSequenceB=conservedProteinSequenceB;
                                          
POTest=[];
NETest=[];
for i=1:length(posTestI)
    POTest = [POTest; 
        AFisherM0Vectors(posTestI(i), :) AFisherM1Vectors(posTestI(i), :) ...
        getAAindexVector(conservedProteinSequenceA, posTestI(i), 5) ...
        BFisherM0Vectors(posTestJ(i), :) BFisherM1Vectors(posTestJ(i), :)...
        getAAindexVector(conservedProteinSequenceB, posTestJ(i), 5) 1];
end
%%%%%%% get the index of NETestI %%%%%%%%%%%%
%Randomly pick same number of negtive as positive.    
numPos = length(posTestI);
numNeg = length(negTestI);
%numNeg = numPos;
%r=randperm(length(posTestI));
%r=r(1:numPos);
%indTestNE=sort(r);

%for i=1:length(indTestNE)
for i=1:length(negTestI)
NETest = [NETest;
    AFisherM0Vectors(negTestI(i), :) AFisherM1Vectors(negTestI(i), :)... 
    getAAindexVector(conservedProteinSequenceA, negTestI(i), 5) ...
    BFisherM0Vectors(negTestJ(i), :) BFisherM1Vectors(negTestJ(i), :)...
    getAAindexVector(conservedProteinSequenceB, negTestJ(i), 5) 0];
end

V=[POTest;NETest];

                    
return;
