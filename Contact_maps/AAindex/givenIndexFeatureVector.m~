function featureV=givenIndexFeatureVector(sequence, index)
% function that return the feature vectore at given index of an amino acid.
load aaIndex;
aa=sequence(index);
cmd=['aaIndex.' aa];
% adding 17 chemicalPhysics features.
featureV=eval(cmd);
end
