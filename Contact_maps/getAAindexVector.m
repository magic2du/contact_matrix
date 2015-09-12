function V=getAAindexVector(seqence, location, WINDOW)
%compute the aaIndex vector for aa in protein. ...
% seqence is the protein sequence, indices is those index you we want to get AAindex
% If WINDOW is ture, you will get this size sliding window around the AA,
% if location =10 WINDOW=5 indices will be 5     6     7     8     9    10    11    12    13    14    15
V=[];
indices=(location-WINDOW):(location+WINDOW);

load aaIndex.mat; %load the aaIndex struct and access aaIndex.A or aaIndex.C
for idxCtr = 1:length(indices)
    if indices(idxCtr)>length(seqence) | indices(idxCtr)<1
        tmpV=zeros(17,1);
    else
        AA=seqence(indices(idxCtr));
        if AA=='-'
            tmpV=zeros(17,1);
        else
            cmd=['aaIndex.' AA];
            tmpV=eval(cmd);
        end
    end
    V=[V tmpV'];
end

end
