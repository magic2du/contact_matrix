function [trainData_scale, testData_scale]=preProcessData(trainData, testData)
    %trainDatawt= trainDatawt ./2 +0.5;
    %testDatawt= bsxfun(@minus, testData, mu);
    %testDatawt=testDatawt*whMat;
    %testDatawt= testDatawt ./2 +0.5;
    %just scale feature to 0-1
    wholeData=[trainData; testData];
    [wholeData_scale, minval, range] = scale_0_1(wholeData);
    trainData_scale= (trainData- repmat(minval, size(trainData, 1) ,1)) ./repmat(range, size(trainData, 1) ,1);
    testData_scale= (testData- repmat(minval, size(testData, 1) ,1)) ./repmat(range, size(testData, 1) ,1);
end
