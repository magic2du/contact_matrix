function build_contactMaps_2011(ddiName)


    
    % load the ddi structure.
    folder3did = ['/home/du/Protein_Protein_Interaction_Project/3did_Apr2011/dom_dom_ints/'];
    ddiPath = [folder3did ddiName '/'];
    ddiStructFile = [ddiPath 'ddi_str_array.mat'];
    load(ddiStructFile);
    
    % read HMM structures for the two domains involved in the interaction.
    domA = ddi_str_array{1}.domainA;
    domB = ddi_str_array{1}.domainB;
    hmmA = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/PFAM/Pfam26Nov2011/SINGLE_FILES/' domA '.pfam']);
    hmmB = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/PFAM/Pfam26Nov2011/SINGLE_FILES/' domB '.pfam']);
                        

    % draw a contact map for each interacting sequence paper, and calculate
    % their averages on the fly.
    % matrix only for the match states of the model
    contactMapAv = zeros(hmmA.ModelLength, hmmB.ModelLength);
    contactMapsBag = {};
    %for pairCtr = 1:length(pairsToRun)
    for seqCtr = 1:length(ddi_str_array)
        
        %seqCtr = pairsToRun(pairCtr);
        
        seqA = ddi_str_array{seqCtr}.ASequence;
        seqB = ddi_str_array{seqCtr}.BSequence;

        seqAResNum = ddi_str_array{seqCtr}.AResNum;ddiPath
        seqBResNum = ddi_str_array{seqCtr}.BResNum;
        
        % align each sequence to their corresponding HMM.
        [scoreA, algnA] = hmmprofalign(hmmA, seqA, 'flanks', true);
        indMatchInsA = regexp(algnA, '[a-zA-Z]');
        indDelA = strfind(algnA, '-');
        indMatchDelA = union(indDelA, regexp(algnA, '[A-Z]'));
        algnAResNum = [];
        algnAResNum(indDelA) = -Inf;
        algnAResNum(indMatchInsA) = seqAResNum;
        hmmAResNum = algnAResNum(indMatchDelA);
        %original sequence number aligned to HMM
        hmmASeq = algnA(indMatchDelA);

        [scoreB, algnB] = hmmprofalign(hmmB, seqB, 'flanks', true);
        indMatchInsB = regexp(algnB, '[a-zA-Z]');
        indDelB = strfind(algnB, '-');
        indMatchDelB = union(indDelB, regexp(algnB, '[A-Z]'));
        algnBResNum = [];
        algnBResNum(indDelB) = -Inf;
        algnBResNum(indMatchInsB) = seqBResNum;
        hmmBResNum = algnBResNum(indMatchDelB);
        hmmBSeq = algnB(indMatchDelB);
        
        % have to draw a dot in the contact map for each pair of
        % interacting residues reported by 3DID.
        intResA = ddi_str_array{seqCtr}.interactA;
        intResB = ddi_str_array{seqCtr}.interactB;

        contactMap = zeros(hmmA.ModelLength, hmmB.ModelLength);
        for intResCtr = 1:length(intResA)
            resA = intResA{intResCtr};
            resA = str2double(resA(regexp(resA, '[0-9]')));
            resB = intResB{intResCtr};
            resB = str2double(resB(regexp(resB, '[0-9]')));
            contactMapPosA = find(hmmAResNum == resA);
            contactMapPosB = find(hmmBResNum == resB);
            if (~isempty(contactMapPosA)) && (~isempty(contactMapPosB))
                contactMap(contactMapPosA, contactMapPosB) = 1;
            end
        end
        

        % save the matlab arrays.
        contactMapsBag(end+1) = {contactMap};
        
        
        % average.
        contactMapAv = contactMapAv + contactMap;
        
    end
    
    % save the matlab arrays.
    contactMapsBag(end+1) = {contactMapAv};
    length(contactMapsBag)
    save([ddiPath 'contactMapsBag.mat'], 'contactMapsBag');
    'built contactBag success!'                                                


    
end

