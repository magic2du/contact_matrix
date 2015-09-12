function contactMatrix = runSeqPair_predict_2011 ...
                        (ddiName, SeqA, SeqB)

%function auc = runSeqPairAllVectorsChooseNegRand_02NOV2011()
%{
pairNbr = 17;
ddiName = 'PF00385.16_int_PF00385.16';

%}
FisherMode = 'FisherM1';
SVMMode = 'SVMLIGHT';
Kernel = 'POLY';
folder3did = ['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_20NOV2009/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
if exist(SVMmodelFile, 'file') % if the SVM model exist for this DDI
    contactMatrix = runSeqPair_predict_matrix(ddiName, SeqA, SeqB, FisherMode, SVMMode, Kernel);
else
    runSeqPair_build_matrix_model(ddiName, FisherMode, SVMMode, Kernel);
    contactMatrix = runSeqPair_predict_matrix(ddiName, SeqA, SeqB, FisherMode, SVMMode, Kernel);
end


return;
