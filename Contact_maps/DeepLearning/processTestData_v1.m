function [testData , testLabels]=processTestData(testFilePath, mu, whMat, chooseNegtive);
% get the whitened trainging data and mu (mean), whMat(whitening matrix) used for precess testData, true for random select negative data.
rawData=load(testFilePath);
    if chooseNegtive

        trueIndex=find(rawData(:,end)==1);
        selectedIndex=getSelectedIndex(trueIndex, rawData);
        selectedData=rawData(selectedIndex, :);
        testLabels=selectedData(:,end);
        testDataWithoutWhiten=selectedData(:,1:end-1);
        testData= bsxfun(@minus, testDataWithoutWhiten, mu);
        testData=testData*whMat;
        
    else
        testLabels=rawData(:,end);
        testDataWithoutWhiten=rawData(:,1:end-1);
        testData= bsxfun(@minus, testDataWithoutWhiten, mu);
        testData=testData*whMat;
    end
end

function selectedIndex=getSelectedIndex(trueIndex, rawData)
    selectedIndex=[];
    positaveStart=1;
    positiveEnding=1;
    negStart=1;
    negEnding=1;
    for i= 2:length(trueIndex)
        trueIndex(i);
        if trueIndex(i)-trueIndex(i-1)~=1
            positiveEnding=trueIndex(i-1);
            posLenght=positiveEnding-positaveStart+1;
            selectedIndex=[selectedIndex [positaveStart: trueIndex(i-1)]];
            negStart=trueIndex(i-1)+1;
            negEnding=trueIndex(i)-1;
            negLength=negEnding-negStart+1;
            neg=[negStart:negEnding];
            tmp=randperm(negLength);
            selectedNeg=neg(tmp(1:posLenght));
            selectedIndex=[selectedIndex selectedNeg];
            positaveStart=trueIndex(i);
            if i==length(trueIndex)
                negEnding=length(rawData(:,end));
                selectedIndex=[selectedIndex [positaveStart: trueIndex(end)]];
                posLenght=trueIndex(end)-positaveStart+1;
                negStart=trueIndex(end)+1;
                negLength=negEnding-negStart+1;
                neg=[negStart:negEnding];
                tmp=randperm(negLength);
                selectedNeg=neg(tmp(1:posLenght));
                selectedIndex=[selectedIndex selectedNeg];
            end
            
        elseif i==length(trueIndex)    
            negEnding=length(rawData(:,end));
            selectedIndex=[selectedIndex [positaveStart: trueIndex(end)]];
            posLenght=trueIndex(end)-positaveStart+1;
            negStart=trueIndex(end)+1;
            negLength=negEnding-negStart+1;
            neg=[negStart:negEnding];
            tmp=randperm(negLength);
            selectedNeg=neg(tmp(1:posLenght));
            selectedIndex=[selectedIndex selectedNeg];
            
        end
    end
end
