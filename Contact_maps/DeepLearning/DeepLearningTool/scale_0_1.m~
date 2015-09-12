function [trainData_scale, minval, range] = scale_0_1(trainData)
minval=min(trainData);
maxval=max(trainData);
range=maxval-minval;

trainData_scale= (trainData- repmat(minval, size(trainData, 1) ,1)) ./repmat(range, size(trainData, 1) ,1);

end

