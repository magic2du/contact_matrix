function ddi_seqs_and_intvectors_topology_level_remote(ddi_folder)

% load the ddi struct.
folder_3did = '/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/';
folder_ddi = [folder_3did ddi_folder '/'];
TopologyListFile=[folder_ddi 'topologyList.txt'];
%topologyList = The topolgy list in the topolgyListFile;
fid = fopen(TopologyListFile, 'r');
topologyList = textscan(fid, '%s', 'delimiter', '\n');
topologyList = topologyList{1};
fclose(fid);
%loop throught topology List
for topology = 1:length(topologyList)
	topologyFolder=[folder_ddi topologyList{topology} '/'];

	ddi_str_file = [topologyFolder '/ddi_str_array.mat'];
	load(ddi_str_file);

	folderPDB = '/home/du/Protein_Protein_Interaction_Project/PDB/';

	ddi_str_array_new = {};
	domASeqs = {};
	domBSeqs = {};
	domALongSeq = '';
	domBLongSeq = '';
	% retrieve sequences from PDB file.
	startTime = clock;
	for pdb_ctr = 1:length(ddi_str_array)
	    
	    pdbstruct = pdbread([folderPDB ddi_str_array{pdb_ctr}.pdbid '.pdb']);
	    [Achain r] = strtok(ddi_str_array{pdb_ctr}.domainApdb, ':');
	    if length(strfind(r, '-')) > 1
		error('pdbstruct has negative amino acid indices');
	    end
	    [Astart_res r] = strtok(r(2:end), '-');
	    Astart_res(regexp(Astart_res, '[A-Za-z]')) = []; % remove letters from the coordinates.
	    Astart_res = str2double(Astart_res);
	    Aend_res = r(2:end);
	    Aend_res(regexp(Aend_res, '[A-Za-z]')) = []; % remove letters from the coordinates.
	    Aend_res = str2double(Aend_res);
	    if Astart_res > Aend_res
		error('Astart_res is greater than Aend_res, cannot parse the sequence');
	    end
	    
	    %try
	    %[seqA,resNumA] = ...
	    %    get_seq_from_struct(pdbstruct, Achain, Astart_res, Aend_res);
	    [seqA,resNumA] = ...
		getSeqFromStruct_24NOV2011(pdbstruct, Achain, Astart_res, Aend_res);
	    if length(seqA) ~= length(resNumA)
		error('seqA and resNumA do not have the same length');
	    end
	    %{
	    catch exception
	    % there was a problem reading the pdb file, won't use this seq.
	    logline = ...
	    ['WARNING: ' ddi_folder ', PDB: ' ...
	    ddi_str_array{pdb_ctr}.pdbid ' ' ddi_str_array{pdb_ctr}.domainApdb ...
	    ', Message: ' exception.message '\n'];
	    fprintf(logline);
	    continue;
	    end
	     %}

	    [Bchain r] = strtok(ddi_str_array{pdb_ctr}.domainBpdb, ':');
	    if length(strfind(r, '-')) > 1
		error('pdbstruct has negative amino acid indices');
	    end
	    [Bstart_res r] = strtok(r(2:end), '-');
	    Bstart_res(regexp(Bstart_res, '[A-Za-z]')) = []; % remove letters from the coordinates.
	    Bstart_res = str2double(Bstart_res);
	    Bend_res = r(2:end);
	    Bend_res(regexp(Bend_res, '[A-Za-z]')) = []; % remove letters from the coordinates.
	    Bend_res = str2double(Bend_res);
	    if Bstart_res > Bend_res
		error('Bstart_res is greater than Bend_res, cannot parse the sequence');
	    end
	    
	    %try
	    %[seqB,resNumB] = ...
	    %    get_seq_from_struct(pdbstruct, Bchain, Bstart_res, Bend_res);
	    [seqB,resNumB] = ...
		getSeqFromStruct_24NOV2011(pdbstruct, Bchain, Bstart_res, Bend_res);
	    if length(seqB) ~= length(resNumB)
		error('seqB and resNumB do not have the same length');
	    end
	    %{
	    catch exception
	    % there was a problem reading the pdb file, won't use this seq.
	    logline = ...
	    ['WARNING: ' ddi_folder ', PDB: ' ...
	    ddi_str_array{pdb_ctr}.pdbid ' ' ddi_str_array{pdb_ctr}.domainBpdb ...
	    ', Message: ' exception.message '\n'];
	    fprintf(logline);
	    continue;
	    end
	    %}

	    ddi_str_array_new(end+1) = {ddi_str_array{pdb_ctr}};
	    ddi_str_array_new{end}.ASequence = seqA;
	    ddi_str_array_new{end}.BSequence = seqB;
	    ddi_str_array_new{end}.AResNum = resNumA;
	    ddi_str_array_new{end}.BResNum = resNumB;
	    
	    % create interaction (binary) vectors. 
	    AInt = cell2vect(ddi_str_array{pdb_ctr}.interactA);
	    [t, ia, ib] = intersect(resNumA, AInt);
	    binaryA = zeros(1, length(resNumA));
	    binaryA(ia) = 1;
	    ddi_str_array_new{end}.InteractionVectorA = binaryA;

	    BInt = cell2vect(ddi_str_array{pdb_ctr}.interactB);
	    [t, ia, ib] = intersect(resNumB, BInt);
	    binaryB = zeros(1, length(resNumB));
	    binaryB(ia) = 1;
	    ddi_str_array_new{end}.InteractionVectorB = binaryB;
	    
	    % data struct to later print sequences to fasta file.
	    seqA = lower(seqA);
	    seqA(binaryA == 1) = upper(seqA(binaryA == 1));
	    domASeqs(end+1).Sequence = seqA;
	    domASeqs(end).Header = ...
	    [ddi_str_array{pdb_ctr}.pdbid ' ' ddi_str_array{pdb_ctr}.domainApdb];

	    seqB = lower(seqB);
	    seqB(binaryB == 1) = upper(seqB(binaryB == 1));
	    domBSeqs(end+1).Sequence = seqB;
	    domBSeqs(end).Header = ...
	    [ddi_str_array{pdb_ctr}.pdbid ' ' ddi_str_array{pdb_ctr}.domainBpdb];

	    % concat sequences to later calculate aa distributions for each domain.
	    domALongSeq = [domALongSeq upper(seqA)];
	    domBLongSeq = [domBLongSeq upper(seqB)];
	    
	    % if this is taking too long, go on!
	    eTime = etime(clock, startTime);
	    if eTime > 120*60
		error('This DDI was taking too long.');
	    end
	    
	end

	% write fasta files.
	fasta_fileA = [topologyFolder '/domASeqs.fasta'];
	if exist(fasta_fileA, 'file')
	    system(['rm -f ' fasta_fileA]);
	end
	try
	fastawrite(fasta_fileA, domASeqs);
	catch
	    breakPoint = 1;
	end
	fasta_fileB = [topologyFolder '/domBSeqs.fasta'];
	if exist(fasta_fileB, 'file')
	    system(['rm -f ' fasta_fileB]);
	end
	try
	fastawrite(fasta_fileB, domBSeqs);
	catch
	    breakPoint = 1;
	end

	% aa distributions for each domain.
	domA_AADistrib = aacount(domALongSeq);
	domA_AADistFile = [topologyFolder '/domA_AADistrib'];
	save(domA_AADistFile, 'domA_AADistrib');
	domB_AADistrib = aacount(domBLongSeq);
	domB_AADistFile = [topologyFolder '/domB_AADistrib'];
	save(domB_AADistFile, 'domB_AADistrib');

	% save updated ddi_str.
	clear ddi_str_array;
	ddi_str_array = ddi_str_array_new;
	save(ddi_str_file, 'ddi_str_array');
end
return;


function vect = cell2vect(cell_array)

% the numbers stored in interact[A][B] are in reality strings. They might
% contain characters, in which case we'll disregard them an use only the
% numeric part of the string.

vect = [];
for resCtr = 1:length(cell_array)
    resNum = cell_array{resCtr};
    resNum(abs(resNum) > 57) = '';  % funct. abs gives you the ascii code.
    vect(end+1) = str2double(resNum);
end

return;
