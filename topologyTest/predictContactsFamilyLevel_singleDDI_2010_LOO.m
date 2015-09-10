function         [SimilarityVector, recallVector, specificityVector, precisionVector, accuracyVector, F1Vector, MCCVector, scoreVector]=predictContactsFamilyLevel_singleDDI_2010_LOO(ddi_folder,ddi_str_array)

% calculate family's cosin similary between interaction vectors and predicted intervector.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currDir = pwd;
cd /home/du/Protein_Protein_Interaction_Project/ipHMMs;

% domainA.
%%%%%%%%%%
SimilarityVector=[];
recallVector=[];
specificityVector=[];
precisionVector=[];
accuracyVector=[];
F1Vector=[];
MCCVector=[];
scoreVector=[];
for i = 1:length(ddi_str_array)
    
    if isempty(ddi_str_array{i}.ASequence)
        error('ASequence is empty');
    end
    
    % if the sequence and its interaction binary vector don't have the same
    % length, there was an error in RetrSeqs_BuildBinIntVect, therefore the
    % ddi can't be used.
    if length(ddi_str_array{i}.ASequence) ~= length(ddi_str_array{i}.InteractionVectorA)
        error('sequence and interaction vector do not have the same length');
    end
    %try
	load([ddi_folder '/iphmmA' num2str(i) '.mat']);
    load([ddi_folder '/iphmmB' num2str(i) '.mat']);
    typeofMatchA=[];
    typeofMatchB=[];
    % get predicted vecotr A
    [scoreA, alignment, path, typeofMatchA, flags] = ...
                                    ipHMMProfAlign(iphmmA,ddi_str_array{i}.ASequence, 'flanks',true);
    typeofMatchA=typeofMatchA'; %transpose typeofMatch '-1': Non interacting match; 1: interacting match, 0: deletes and insert. it ingore 0's from both side, so lenght=168 instead of 217, alignment length, 
    predVectorA = typeofMatchA(typeofMatchA ~= 0); % delete zeros length=130, 1more for insert.
    alignedSequenceA=strrep(alignment,'-',''); % delete - from aglignment lenth=131
    lowerCaseLocationA=regexp(alignedSequenceA,'[a-z]'); %find the location of lower case because it is not in the typeofMatch
    if length(lowerCaseLocationA)==1
        predVectorA = [predVectorA(1:lowerCaseLocationA-1) 0 predVectorA(lowerCaseLocationA:length(predVectorA))]; 
    else
        for j=1:length(lowerCaseLocationA)
            predVectorA = [predVectorA(1:lowerCaseLocationA(j)-1) 0 predVectorA(lowerCaseLocationA(j):length(predVectorA))];
        end
    end
    % change -1 to 0 in predVector
    predVectorA(find(predVectorA==-1))=0;
    % end get predicte vector A
    [recallA,specificityA,precisionA,accuracyA,F1A,MCCA]=predictionMeasure(predVectorA, ddi_str_array{i}.InteractionVectorA);
    %get preditec vector B
        [scoreB, alignment, path, typeofMatchB, flags] = ...
                                    ipHMMProfAlign(iphmmB,ddi_str_array{i}.BSequence, 'flanks',true);
    typeofMatchB=typeofMatchB'; %transpose typeofMatch '-1': Non interacting match; 1: interacting match, 0: deletes and insert. it ingore 0's from both side, so lenght=168 instead of 217, alignment length, 
    predVectorB=typeofMatchB(typeofMatchB ~= 0);% delete zeros length=130, 1more for insert.
    alignedSequenceB=strrep(alignment,'-',''); % delete - from aglignment lenth=131
    lowerCaseLocationB=regexp(alignedSequenceB,'[a-z]'); %find the location of lower case because it is not in the typeofMatch
    if length(lowerCaseLocationB)==1
        predVectorB = [predVectorB(1:lowerCaseLocationB-1) 0 predVectorB(lowerCaseLocationB:length(predVectorB))] ;
    else
        for j=1:length(lowerCaseLocationB)
            predVectorB = [predVectorB(1:lowerCaseLocationB(j)-1) 0 predVectorB(lowerCaseLocationB(j):length(predVectorB))];
        end
    end
    % change -1 to 0 in predVector
    predVectorB(find(predVectorB==-1))=0;
     %end get preditec vector B
    [recallB,specificityB,precisionB,accuracyB,F1B,MCCB]=predictionMeasure(predVectorB, ddi_str_array{i}.InteractionVectorB);
    
     jointPredVector=[predVectorA predVectorB];
     jointTureVector=[ddi_str_array{i}.InteractionVectorA ddi_str_array{i}.InteractionVectorB];
        
    % get the cosine similarity and other average measurement
    SimilarityVector(i)=cosine_similarity(jointPredVector,jointTureVector);
    recallVector(i)=(recallA+recallB)/2;
    specificityVector(i)=(specificityA+specificityB)/2;
    precisionVector(i)=(precisionA+precisionB)/2;
    accuracyVector(i)=(accuracyA+accuracyB)/2;
    F1Vector(i)=(F1A+F1B)/2;
    MCCVector(i)=(MCCA+MCCB)/2;
    scoreVector(i)=(scoreA+scoreB)/2;
    
    
end

cd(currDir);
        
return;
