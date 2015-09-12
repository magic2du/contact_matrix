function score_contactMatrix= predict_contact_matrix_with_SVMmodel_2011(ddiName, SeqA, SeqB, FisherMode, SVMMode, Kernel, SVMmodelFile, tsvPath)
% calculate the score_contactMatrix for given squence pair SeqA and SeqB.

folder3did = ['/home/du/Protein_Protein_Interaction_Project/3did_Apr2011/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
folderResults = ddiPath;



% read HMM structures for the two domains involved in the interaction.
load([ddiPath 'ddi_str_array.mat']); % AFisherM0[1]Array, AconstFisherM0[1]Array.
domA = ddi_str_array{1}.domainA;
domB = ddi_str_array{1}.domainB;
hmmA = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/PFAM/Pfam26Nov2011/SINGLE_FILES/' domA '.pfam']);
hmmB = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/PFAM/Pfam26Nov2011/SINGLE_FILES/' domB '.pfam']);

%compute the HMM aligned sequence ignoring insert, should be HMM length
seqA = SeqA;
seqB = SeqB;
% seqA
[scoreA, algnA] = hmmprofalign(hmmA, seqA, 'flanks', true);
indDelA = strfind(algnA, '-');
indMatchDelA = union(indDelA, regexp(algnA, '[A-Z]'));
conservedProteinSequenceA = algnA(indMatchDelA); 
% seqB
[scoreB, algnB] = hmmprofalign(hmmB, seqB, 'flanks', true);
indDelB = strfind(algnB, '-');
indMatchDelB = union(indDelB, regexp(algnB, '[A-Z]'));
conservedProteinSequenceB = algnB(indMatchDelB); 

% Fisher vectors of query sequence pair. We will use Fisher M1 model
load([ddiPath 'iphmmA.mat']);
load([ddiPath 'iphmmB.mat']);
currDir = pwd;
cd /home/du/Protein_Protein_Interaction_Project/ipHMMs;
[AFisherM0 AFisherM1 AconstFisherM0 AconstFisherM1] = ...
                                    calculateFisherVector(seqA, iphmmA);
[BFisherM0 BFisherM1 BconstFisherM0 BconstFisherM1] = ...
                                    calculateFisherVector(seqB, iphmmB);
cd(currDir);
ATestFisherVectors = eval(['A' FisherMode]);
BTestFisherVectors = eval(['B' FisherMode]);
% Testing.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
score_contactMatrix = zeros(hmmA.ModelLength, hmmB.ModelLength);

POTest=[];
NETest=[];

for i=1:hmmA.ModelLength
    for j=1:hmmB.ModelLength
        POTest = [POTest; ...
        ATestFisherVectors(i, :) BTestFisherVectors(j, :) ];
    end
end


size(POTest)
if strcmp(SVMMode, 'SVMLIGHT')

% test on test set:
testFile = ...
    [folderResults FisherMode '_'  SVMMode '_' Kernel '_'  '_build_matrix_model_with_contactMaps.test'];
%resultFile = [folderResults 'AllVectorsChooseNegRand_02NOV2011_' ...
 %   FisherMode '_' SVMMode '_' Kernel '_pair' num2str(pairNbr) '.result'];
resultFile = [folderResults ...
    FisherMode '_'  SVMMode '_' Kernel '_' 'tmp_test.result'];

write_SVM_file(POTest, NETest, testFile);
svmlightFolder='/home/du/Protein_Protein_Interaction_Project/svm_light_linux64_2013/';
command = ...
[svmlightFolder 'svm_classify ' testFile ' ' SVMmodelFile ' ' resultFile];
system(command);
testResult = load(resultFile);
score_contactMatrix=reshape(testResult, hmmB.ModelLength, hmmA.ModelLength);
score_contactMatrix=score_contactMatrix';
matrixFilePath=[tsvPath 'scoreContactMatrix.tsv'];
%save(matrixFilePath, 'score_contactMatrix', '-ascii','-tabs');
savetsv(matrixFilePath, score_contactMatrix);
end

return;
