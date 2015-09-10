function [iphmmA, iphmmB] = generate_iphmm(ddi_struct, phmmA, phmmB)

% April 14, 2011
% This function was copied from the svn repository, repos/iphmm-etb, which
% is Colin's early trace back project. Only had to change the path to the
% ipHMMs folder.

ipHMMDir = '/home/du/Protein_Protein_Interaction_Project/ipHMMs/';

% Uses a single domain-domain interaction structure and a phmm to estimate
% the iphmm.

% Output - two iphmms for the two interacting domains, A and B.

% First work domainA. Need to create a multiple sequence alignmet by
% aligning all the domainA sequences to the phmmA. These aligned seqs are
% stored in a new structure under the entry 'Sequence'. The interaction
% vector of 0's and 1's for these sequences are also copied over to the
% newly created structure under the entry 'Interactions'.

% domainA.
ASequenceStruct = struct([]);
AInteractionStruct = struct([]);

%fprintf('Aligning sequences for domainA ');

% 09/23/2010.
% Need to change this for loop. Tapan is not making sure all the sequences
% and interacting vectors are the same length (should be the domain
% length).

for i = 1:length(ddi_struct)
    %fprintf('.')
    [score, aligned_seq] = hmmprofalign(phmmA, ddi_struct{i}.ASequence, 'FLANKS', true);
    ASequenceStruct(i).Sequence = aligned_seq;
    Deletes = strfind(aligned_seq, '-');
    aligned_int = [];
    aligned_int(Deletes) = -1;
    MatchInserts = regexp(aligned_seq, '[A-Za-z]');
    aligned_int(MatchInserts) = ddi_struct{i}.InteractionVectorA;
    AInteractionStruct(i).interactions = ...
        find(aligned_int == 1);%ddi_struct{i}.InteractionVectorA;
end

% for i = 1:length(ddi_struct)
%     %fprintf('.')
%     [score, aligned_seq] = hmmprofalign(phmmA, ddi_struct{i}.ASequence);
%     Deletes = strfind(aligned_seq, '-');
%     aligned_int = [];
%     aligned_int(Deletes) = -1;
%     MatchInserts = regexp(aligned_seq, '[A-Za-z]');
%     aligned_int(MatchInserts) = ddi_struct{i}.InteractionVectorA;
%     MatchDeletes = regexp(aligned_seq, '[A-Z-]');
%     domain_seq = aligned_seq(MatchDeletes);
%     domain_int = aligned_int(MatchDeletes);
%     % make delete positions be non-interacting.
%     domain_int(domain_int < 0) = 0;
%     ASequenceStruct(i).Sequence = domain_seq;
%     AInteractionStruct(i).interactions = domain_int;
% end
%fprintf('\n');

% Now estimate iphmm.
%fprintf('Estimating iphmmA \n')
currDir = pwd;
cd(ipHMMDir);
iphmmA = iphmmprofestimate(phmmA, ASequenceStruct, AInteractionStruct);
cd(currDir);

% Same for domainB.
BSequenceStruct = struct([]);
BInteractionStruct = struct([]);

%fprintf('Aligning sequences for domainB ');

% 09/23/2010.
% Need to change this for loop. Tapan is not making sure all the sequences
% and interacting vectors are the same length (should be the domain
% length).

for i = 1:length(ddi_struct)
    %fprintf('.');
    [score, aligned_seq] = hmmprofalign(phmmB, ddi_struct{i}.BSequence, 'FLANKS', true);
    BSequenceStruct(i).Sequence = aligned_seq;
    Deletes = strfind(aligned_seq, '-');
    aligned_int = [];
    aligned_int(Deletes) = -1;
    MatchInserts = regexp(aligned_seq, '[A-Za-z]');
    try
        aligned_int(MatchInserts) = ddi_struct{i}.InteractionVectorB;
    catch exp
        colin = 1;
    end

    BInteractionStruct(i).interactions = ...
        find(aligned_int == 1);%ddi_struct{i}.InteractionVectorB;
end
%{
for i = 1:length(ddi_struct)
    %fprintf('.')
    [score, aligned_seq] = hmmprofalign(phmmB, ddi_struct{i}.BSequence);
    Deletes = strfind(aligned_seq, '-');
    aligned_int = [];
    aligned_int(Deletes) = -1;
    MatchInserts = regexp(aligned_seq, '[A-Za-z]');
    aligned_int(MatchInserts) = ddi_struct{i}.InteractionVectorB;
    MatchDeletes = regexp(aligned_seq, '[A-Z-]');
    domain_seq = aligned_seq(MatchDeletes);
    domain_int = aligned_int(MatchDeletes);
    % make delete positions be non-interacting.
    domain_int(domain_int < 0) = 0;
    BSequenceStruct(i).Sequence = domain_seq;
    BInteractionStruct(i).interactions = domain_int;
end
%}
%fprintf('\n');

% Now estimate iphmm.
%fprintf('Estimating iphmmB \n')
cd(ipHMMDir);
iphmmB = iphmmprofestimate(phmmB, BSequenceStruct, BInteractionStruct);
cd(currDir);

return;
