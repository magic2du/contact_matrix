function ContactMapsOn3DID_2010_new(file_ddis)

diary off;
%logfile_name = 'log_ContactMapsOn3DID_29OCT2010.txt';
%logfile_name = 'log_ContactMapsOn3DID_09AUG2011.txt';
%logfile_name = 'log_ContactMapsOn3DID_01NOV2011.txt';
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_ContactMapsOn3DID_2010_' dateFormatted '.txt'];
%logfile_name = 'log_ContactMapsOn3DID_05NOV2011.txt';
if exist(logfile_name, 'file')
    command = ['rm -f ' logfile_name];
    system(command);
end
diary(logfile_name);

if ~exist('ContactMapExamples', 'dir')
    mkdir('ContactMapExamples');
end

% we'll only do the ddis in this dataset.
%file_ddis = 'dataset_01JAN2010.txt';
%file_ddis = 'dataset_09AUG2011.txt';
%file_ddis = 'finishedDDIs_GraphClustering_31OCT2011.txt';
%file_ddis = 'showcaseDDI_05NOV2011.txt';
fid = fopen(file_ddis, 'r');
cell_ddis = textscan(fid, '%s', 'delimiter', '\n');
cell_ddis = cell_ddis{1};
fclose(fid);

for ddi_ctr = 1:length(cell_ddis)
    diary off;
    diary on;
    try
    %{
    ddi_line = textscan(cell_ddis{ddi_ctr}, '%s %s');
    domAPfam = ddi_line{1}{1};
    domBPfam = ddi_line{2}{1};
    ddi_folder = [domAPfam '_int_' domBPfam];
    %}
    ddi_folder = cell_ddis{ddi_ctr};
    
    % create folder to save all contact maps in this ddi family.
    if ~exist(['ContactMapExamples/' ddi_folder], 'dir')
        mkdir(['ContactMapExamples/' ddi_folder]);
    end
    
    % load the ddi structure.
    folder3did = ['/big/du/Protein_Protein_Interaction_Project/3did_15OCT2010_new_whole/dom_dom_ints/'];
    ddiPath = [folder3did ddi_folder '/'];
    ddiStructFile = [ddiPath '/ddi_str_array.mat'];
    load(ddiStructFile);
    
    % read HMM structures for the two domains involved in the interaction.
    domA = ddi_str_array{1}.domainA;
    domB = ddi_str_array{1}.domainB;
    hmmA = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/' ...
                            'PFAM_2008/SINGLE_FILES/' domA '.pfam']);
    hmmB = pfamhmmread(['/home/du/Protein_Protein_Interaction_Project/' ...
                            'PFAM_2008/SINGLE_FILES/' domB '.pfam']);
                        

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

        seqAResNum = ddi_str_array{seqCtr}.AResNum;
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
        
        % draw and save the contact map for this pair of interacting
        % sequences.
        
%        figure; imshow(contactMap);
%        fileName = ['ContactMapExamples/' ddi_folder ...
%                                   '/contactMap' num2str(seqCtr) '.pdf'];
%        print('-dpdf', fileName);
%        close all;
        
        
        % save the matlab arrays.
        contactMapsBag(end+1) = {contactMap};
        
        % save pretty printed text files.
        fileName = ['ContactMapExamples/' ddi_folder ...
                                   '/contactMap' num2str(seqCtr) '.txt'];
        vertHeader = hmmASeq;
        horizHeader = hmmBSeq;
        prettyPrintMatrix(contactMap, true, fileName, ...
                                            horizHeader, vertHeader, '\t');
        
        % average.
        contactMapAv = contactMapAv + contactMap;
        
    end
    
    % save the matlab arrays.
    contactMapsBag(end+1) = {contactMapAv};
    save(['ContactMapExamples/' ddi_folder '/contactMapsBag.mat'], ...
                                                        'contactMapsBag');
                                                    
    % save pretty printed text files for average map.
    fileName = ['ContactMapExamples/' ddi_folder '/contactMapAv.txt'];
    vertHeader = '';
    horizHeader = '';
    prettyPrintMatrix(contactMapAv, true, fileName, ...
                                            horizHeader, vertHeader, '\t');
    
    % scale contactMapAv to [0-1].
    contactMapAv = (1/max(contactMapAv(:)))*contactMapAv;
    
    % draw and save the average contact map for the whole ddi family.
    %{
    figure; imshow(contactMapAv);
    fileName = ['ContactMapExamples/' ddi_folder '/contactMapAv.pdf'];
    print('-dpdf', fileName);
    close all;
    %}
    
    % report success.
    fprintf(['ddi ' ddi_folder ': success!\n']);
    
    catch exc
        close all;
        fprintf(['ddi ' ddi_folder ': error\n']);
        
        % print the stack.
        for stackCtr = length(exc.stack):-1:1
            fprintf([exc.stack(stackCtr).file '; ' ...
                        exc.stack(stackCtr).name '; ' ...
                        num2str(exc.stack(stackCtr).line) '\n']);
        end
    end
    
end

diary off;

return;
