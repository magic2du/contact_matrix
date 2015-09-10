function [iphmmA iphmmB ...
            AFisherM0Array AFisherM1Array ...
            AconstFisherM0Array AconstFisherM1Array ...
            MSA_domA ConsSeqA int_sitesA ...
            BFisherM0Array BFisherM1Array ...
            BconstFisherM0Array BconstFisherM1Array ...
            MSA_domB ConsSeqB int_sitesB] = ...
                        runDDIFamily_IphmmFisher_topology_level_remote(ddi_str_array)

% calculate family's iphmms.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

currDir = pwd;
cd /home/du/Protein_Protein_Interaction_Project/ipHMMs;

% domainA.
%%%%%%%%%%
ASequenceStruct = struct([]);
AInteractionStruct = struct([]);
MSA_domA = {};
int_sitesA = [];
phmmA_name = ddi_str_array{1}.domainA;
folderPfam = ...
'/home/du/Protein_Protein_Interaction_Project/PFAM_2008/SINGLE_FILES/';
phmmA = pfamhmmread([folderPfam phmmA_name '.pfam']);

for i = 1:length(ddi_str_array)
    
    if isempty(ddi_str_array{i}.ASequence)
        error('ASequence is empty');
    end
    
    % if the sequence and its interaction binary vector don't have the same
    % length, there was an error in RetrSeqs_BuildBinIntVect, therefore the
    % ddi can't be used.
    if length(ddi_str_array{i}.ASequence) ~= length(ddi_str_array{i}.InteractionVectorA)
        error('sequence and interaction vector do not have the same length');
    end
    %try
    [score, aligned_seq] = ...
        hmmprofalign(phmmA, ddi_str_array{i}.ASequence, 'FLANKS', true);
    %catch
    %    breakPoint = 1;
    %end
    ASequenceStruct(i).Sequence = aligned_seq;
    Deletes = strfind(aligned_seq, '-');
    aligned_int = [];
    aligned_int(Deletes) = -1;
    MatchInserts = regexp(aligned_seq, '[A-Za-z]');
    %try
    aligned_int(MatchInserts) = ddi_str_array{i}.InteractionVectorA;
    %catch
    %    breakPoint = 1;
    %end
    AInteractionStruct(i).interactions = find(aligned_int == 1);
    
    % create MSA on the fly.
    MatchDeletes = regexp(aligned_seq, '[A-Z-]');
    alignedSeqMSA = aligned_seq(MatchDeletes);
    alignedIntMSA = aligned_int(MatchDeletes);
    alignedSeqMSA(alignedIntMSA == 1) = ...
                                lower(alignedSeqMSA(alignedIntMSA == 1));
    MSA_domA(i).Sequence = alignedSeqMSA;
    MSA_domA(i).Header = ...
                [ddi_str_array{i}.pdbid ' ' ddi_str_array{i}.domainApdb];
    int_sitesA = union(int_sitesA, find(alignedIntMSA == 1));
end

% Now estimate iphmm.
try
iphmmA = iphmmprofestimate(phmmA, ASequenceStruct, AInteractionStruct);
catch
    % the previous function sometimes throws an error when -- I believe --
    % there are inserts in the MSA. I know that if I remove inserts I do
    % get a nice-looking, perfectly rectangular MSA, but I lose information
    % to train insert states. However, this is what we'll do in these
    % cases.
    ASequenceStruct = struct([]);
    AInteractionStruct = struct([]);
    for i = 1:length(ddi_str_array)

        [score, aligned_seq] = ...
        hmmprofalign(phmmA, ddi_str_array{i}.ASequence, 'FLANKS', true);
        indMatch = regexp(aligned_seq, '[A-Z]');
        indInsert = regexp(aligned_seq, '[a-z]');
        indDelete = strfind(aligned_seq, '-');

        intVectorAlgn = zeros(size(aligned_seq));
        intVectorAlgn(union(indMatch, indInsert)) = ...
                                    ddi_str_array{i}.InteractionVectorA;

        ASequenceStruct(i).Sequence = ...
                                aligned_seq(union(indMatch, indDelete));
        AInteractionStruct(i).interactions = ...
                                intVectorAlgn(union(indMatch, indDelete));

    end
    iphmmA = iphmmprofestimate(phmmA, ASequenceStruct, AInteractionStruct);
end

% domainB.
%%%%%%%%%%
BSequenceStruct = struct([]);
BInteractionStruct = struct([]);
MSA_domB = {};
int_sitesB = [];
phmmB_name = ddi_str_array{1}.domainB;
phmmB = pfamhmmread([folderPfam phmmB_name '.pfam']);

for i = 1:length(ddi_str_array)
    
    if isempty(ddi_str_array{i}.BSequence)
        error('BSequence is empty');
    end
    
    % if the sequence and its interaction binary vector don't have the same
    % length, there was an error in RetrSeqs_BuildBinIntVect, therefore the
    % ddi can't be used.
    if length(ddi_str_array{i}.ASequence) ~= length(ddi_str_array{i}.InteractionVectorA)
        error('sequence and interaction vector do not have the same length');
    end
    [score, aligned_seq] = ...
        hmmprofalign(phmmB, ddi_str_array{i}.BSequence, 'FLANKS', true);
    BSequenceStruct(i).Sequence = aligned_seq;
    Deletes = strfind(aligned_seq, '-');
    aligned_int = [];
    aligned_int(Deletes) = -1;
    MatchInserts = regexp(aligned_seq, '[A-Za-z]');
    aligned_int(MatchInserts) = ddi_str_array{i}.InteractionVectorB;
    BInteractionStruct(i).interactions = find(aligned_int == 1);
    
    % create MSA on the fly.
    MatchDeletes = regexp(aligned_seq, '[A-Z-]');
    alignedSeqMSA = aligned_seq(MatchDeletes);
    alignedIntMSA = aligned_int(MatchDeletes);
    alignedSeqMSA(alignedIntMSA == 1) = ...
                                lower(alignedSeqMSA(alignedIntMSA == 1));
    MSA_domB(i).Sequence = alignedSeqMSA;
    MSA_domB(i).Header = ...
                [ddi_str_array{i}.pdbid ' ' ddi_str_array{i}.domainBpdb];
    int_sitesB = union(int_sitesB, find(alignedIntMSA == 1));
end

% Now estimate iphmm.
try
iphmmB = iphmmprofestimate(phmmB, BSequenceStruct, BInteractionStruct);
catch
    % the previous function sometimes throws an error when -- I believe --
    % there are inserts in the MSA. I know that if I remove inserts I do
    % get a nice-looking, perfectly rectangular MSA, but I lose information
    % to train insert states. However, this is what we'll do in these
    % cases.
    BSequenceStruct = struct([]);
    BInteractionStruct = struct([]);
    for i = 1:length(ddi_str_array)

        [score, aligned_seq] = ...
            hmmprofalign(phmmB, ddi_str_array{i}.BSequence, 'FLANKS', true);
        indMatch = regexp(aligned_seq, '[A-Z]');
        indInsert = regexp(aligned_seq, '[a-z]');
        indDelete = strfind(aligned_seq, '-');

        intVectorAlgn = zeros(size(aligned_seq));
        intVectorAlgn(union(indMatch, indInsert)) = ...
                                        ddi_str_array{i}.InteractionVectorB;

        BSequenceStruct(i).Sequence = aligned_seq(union(indMatch, indDelete));
        BInteractionStruct(i).interactions = ...
                                    intVectorAlgn(union(indMatch, indDelete));

    end
    iphmmB = iphmmprofestimate(phmmB, BSequenceStruct, BInteractionStruct);
end

% MSAs and consensus sequences.
ConsSeqA = seqconsensus(MSA_domA);
ConsSeqB = seqconsensus(MSA_domB);

% Fisher vectors for each sequence pair.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AFisherM0Array = {};
AFisherM1Array = {};
AconstFisherM0Array = {};
AconstFisherM1Array = {};
BFisherM0Array = {};
BFisherM1Array = {};
BconstFisherM0Array = {};
BconstFisherM1Array = {};
for seqCtr = 1:length(ddi_str_array)
    
    seqA = ddi_str_array{seqCtr}.ASequence;
    [AFisherM0 AFisherM1 AconstFisherM0 AconstFisherM1] = ...
                                    calculateFisherVector(seqA, iphmmA);
    AFisherM0Array(seqCtr) = {AFisherM0};
    AFisherM1Array(seqCtr) = {AFisherM1};
    AconstFisherM0Array(seqCtr) = {AconstFisherM0};
    AconstFisherM1Array(seqCtr) = {AconstFisherM1};
    
    seqB = ddi_str_array{seqCtr}.BSequence;
    [BFisherM0 BFisherM1 BconstFisherM0 BconstFisherM1] = ...
                                    calculateFisherVector(seqB, iphmmB);
    BFisherM0Array(seqCtr) = {BFisherM0};
    BFisherM1Array(seqCtr) = {BFisherM1};
    BconstFisherM0Array(seqCtr) = {BconstFisherM0};
    BconstFisherM1Array(seqCtr) = {BconstFisherM1};
    
end

cd(currDir);
        
return;
