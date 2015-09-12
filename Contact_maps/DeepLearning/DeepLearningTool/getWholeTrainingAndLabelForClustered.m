function [wholeTrainingData, wholeTrainingLabel]=getWholeTrainingAndLabelForClustered(pairNbr, dataFolder, FisherMode, choseNegRatio)
%%%%%%%%%%%%%%%GET REDUCED SEQUENCE ID%%%%%%%%%%
wholeTrainingData=[];
wholeTrainingLabel=[];

filePath=[dataFolder 'allPairs.txt'];
pairs= load(filePath);

trainIdx = [];
trainNE=[];
trainPO=[];
for clCtr = 1:length(pairs)
    if pairNbr ~= pairs(clCtr)
        trainIdx(end+1) = pairs(clCtr);
    end
end

    for idxCtr = 1:length(trainIdx)
        seqCtr = trainIdx(idxCtr);
        
        % create positive and negative training sets.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\
        dataFile=[dataFolder 'F0_20_F1_20_Sliding_17_11_F0_20_F1_20_Sliding_17_11_ouput_' num2str(seqCtr) '.txt'];
        numPos=load([dataFolder 'numPos_' num2str(seqCtr) '.txt']);
        numNeg=load([dataFolder 'numNeg_' num2str(seqCtr) '.txt']);
        [selectedData, label]=chooseAAIndexVectores(dataFile, FisherMode);
        PO=selectedData(1:numPos, :);
        NE=selectedData(numPos+1: numPos+numNeg, :);

        % chose negatives as in the the ratio default is 1.
        selectedNE=[];
        if choseNegRatio==0
            selectedNE=NE;
        else
            numbNegTrain = size(PO, 1) * choseNegRatio;
            if numbNegTrain>numNeg
                numbNegTrain=numNeg;
            end
            r=randperm(size(NE, 1));
            r=r(1:numbNegTrain);
            indTrainNE=sort(r);
            selectedNE = NE(indTrainNE, :);
        end
        trainNE=[trainNE; selectedNE];
        trainPO=[trainPO; PO];
    end
    wholeTrainingData=[trainPO; trainNE];
    wholeTrainingLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
end
