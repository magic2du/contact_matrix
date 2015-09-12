function [reducedTrainingData, reducedTrainingLabel]=getReducedTrainingAndLabel(reduceRatio, trainIdx, dataFolder, FisherMode, choseNegRatio)
%%%%%%%%%%%%%%%GET REDUCED SEQUENCE ID%%%%%%%%%%
m=length(trainIdx);
reduceN=floor(m/reduceRatio);
if reduceN==0
    reduceN=1;
end
r=randperm(m);
r=r(1:reduceN);
reducedID=trainIdx(r);
trainNE=[];
trainPO=[];
    for idxCtr = 1:length(reducedID)
        seqCtr = reducedID(idxCtr);
        
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
    reducedTrainingData=[trainPO; trainNE];
    reducedTrainingLabel=[ones(size(trainPO, 1), 1); ...
				    zeros(size(trainNE, 1), 1)];
end
