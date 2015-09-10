function [domainSeq aaNumbArray] = ...
            getSeqFromStruct_24NOV2011(protComplex, chain, startAA, endAA)
 
%{
protComplex = pdbstruct;
chain = Bchain;
startAA = Bstart_res;
endAA = Bend_res;
%}

aaNumbArray = [];
domainSeq = '';
for aaCtr = startAA:endAA
    
    % notice that a PDB structure can have more than one model.
    % => http://www.wwpdb.org/documentation/format33/sect9.html
    
    %try
	pointToChain = find(strcmp({protComplex.Model(1).Atom(:).chainID}, chain));
    %catch
    %    breakPoint = 1;
    %end
    pointToResidue = find([protComplex.Model(1).Atom(:).resSeq]==aaCtr);
    pointToResInChain = intersect(pointToChain, pointToResidue);
    if isempty(pointToResInChain)
        continue;
    else
        aaNumbArray(end+1) = aaCtr;
    end
    
    domainSeq = [domainSeq ...
        upper(aminolookup(protComplex.Model(1).Atom(pointToResInChain(1)).resName))];
	
end
        
return;
