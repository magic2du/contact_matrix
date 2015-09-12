function [batchsize, numepochs]=setBatchsizeAndEpochs(numberOfTrain, miniBatchSize, totalIterations)
    m=numberOfTrain; 
    if m<=miniBatchSize
        batchsize=m;
    else
        batchsize=miniBatchSize;
    end
    numberOfIterationEachEpoch=floor(m/batchsize);
    numepochs=floor(totalIterations/numberOfIterationEachEpoch);
end
