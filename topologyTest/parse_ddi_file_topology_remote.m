function parse_ddi_file_topology_remote(ddi)

% read and parse 3did flat file.
folder_3did = '/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/';
folder_ddi = [folder_3did 'dom_dom_ints/' ddi '/'];
TopologyListFile=[folder_ddi 'topologyList.txt'];
%topologyList = The topolgy list in the topolgyListFile;
fid = fopen(TopologyListFile, 'r');
topologyList = textscan(fid, '%s', 'delimiter', '\n');
topologyList = topologyList{1};
fclose(fid);
%loop throught topology List
for topology = 1:length(topologyList)
	topologyFolder=[folder_ddi topologyList{topology} '/']
	file_ddi = [topologyFolder ddi '.3did'];

	fid = fopen(file_ddi, 'r');
	cell_ddi = textscan(fid, '%s', 'delimiter', '\n');
	cell_ddi = cell_ddi{1};
	fclose(fid);

	state = 'start';
	for line = 1:length(cell_ddi)
	    %try
		if strcmp(state, 'start')
		    
		    if strcmp(cell_ddi{line}(1:4), '#=ID')
		        
		        state = 'dom-dom';
		        ddi_str_array = {};
		        ddi_row = textscan(cell_ddi{line}, '%s %s %s %s %s');
		        clear ppi_str;
		        ppi_str.domainA = ddi_row{2}{1};
		        ppi_str.domainB = ddi_row{3}{1};
		        ppi_str.pfamA = ddi_row{4}{1}(2:(length(ddi_row{4}{1})-5));
		        ppi_str.pfamB = ddi_row{5}{1}(1:(length(ddi_row{5}{1})-6));
		        % will remove the version digits from the acc. code.
		        ppi_str.pfamA = strtok(ppi_str.pfamA, '.');
		        ppi_str.pfamB = strtok(ppi_str.pfamB, '.');
		        
		        % WARNING: HAVE TO DOWNLOAD PFAM FILES AT THIS POINT!
		        
		    end
		    
		elseif strcmp(state, 'dom-dom')
		    
		    state = 'prot-prot';
		    ppi_row = textscan(cell_ddi{line}, '%s %s %s %s %s %s');
		    ppi_str.pdbid = ppi_row{2}{1};
		    ppi_str.domainApdb = ppi_row{3}{1};
		    ppi_str.domainBpdb = ppi_row{4}{1};
		    ppi_str.score = ppi_row{5}{1};
		    ppi_str.Zscore = ppi_row{6}{1};
		    ppi_str.interactA = {};
		    ppi_str.interactB = {};
		    ppi_str.intaaA = '';
		    ppi_str.intaaB = '';
		    %ddi_str_array(end+1) = {ppi_str};
		    
		    % download PDB files.
		    if ~exist(['/home/du/Protein_Protein_Interaction_Project/PDB/' ppi_str.pdbid '.pdb'], 'file')
		        %try
		        PDBStruct = getpdb(ppi_str.pdbid, 'ToFile', ...
		                                ['/home/du/Protein_Protein_Interaction_Project/PDB/' ppi_str.pdbid '.pdb']);
		        %catch
		        %log_line = ...
		        %['PDB file not found: ' ppi_str.pdbid '.\n'];
		        %fprintf(logf_id, log_line);
		        %end
		    end
		    
		elseif strcmp(state, 'prot-prot')
		    
		    %if strcmp(cell_ddi{line}(1:4), '#=ID')             
		    state = 'res-res';
		    resres_row = textscan(cell_ddi{line}, '%s %s %s %s %s');
		    aaA = resres_row{1}{1};
		    aaB = resres_row{2}{1};
		    seqResA = resres_row{3}{1};
		    seqResB = resres_row{4}{1};
		    ppi_str.interactA(end+1) = {seqResA};
		    ppi_str.interactB(end+1) = {seqResB};
		    ppi_str.intaaA(end+1) = aaA;
		    ppi_str.intaaB(end+1) = aaB;
		    
		elseif strcmp(state, 'res-res')
		    
		    if strcmp(cell_ddi{line}(1:2), '//')
		        
		        state = 'end-of-ddi';
		        %datasvet_str(end+1) = {ddi_str_array};
		        ddi_str_array(end+1) = {ppi_str};
		        save([topologyFolder '/ddi_str_array'], 'ddi_str_array');
		        %log_line = ...
		        %['Finished with ' num2str(ddi_ctr) ': ' folder_name '.\n'];
		        %fprintf(logf_id, log_line);
		    
		    elseif strcmp(cell_ddi{line}(1:4), '#=3D')
		        
		        state = 'prot-prot';
		        
		        ddi_str_array(end+1) = {ppi_str};
		        clear ppi_str;
		        %{
		        ppi_str.domainA = [];
		        ppi_str.domainB = [];
		        ppi_str.pfamA = [];
		        ppi_str.pfamB = [];
		        %}
		        ppi_str.domainA = ddi_str_array{end}.domainA;
		        ppi_str.domainB = ddi_str_array{end}.domainB;
		        ppi_str.pfamA = ddi_str_array{end}.pfamA;
		        ppi_str.pfamB = ddi_str_array{end}.pfamB;
		        
		        % WARNING: HAVE TO DOWNLOAD PFAM FILES AT THIS POINT!
		        
		        ppi_row = textscan(cell_ddi{line}, '%s %s %s %s %s %s');
		        ppi_str.pdbid = ppi_row{2}{1};
		        ppi_str.domainApdb = ppi_row{3}{1};
		        ppi_str.domainBpdb = ppi_row{4}{1};
		        ppi_str.score = ppi_row{5}{1};
		        ppi_str.Zscore = ppi_row{6}{1};
		        ppi_str.interactA = {};
		        ppi_str.interactB = {};
		        ppi_str.intaaA = '';
		        ppi_str.intaaB = '';
		        
		        % download PDB files.
		        if ~exist(['/home/du/Protein_Protein_Interaction_Project/PDB/' ppi_str.pdbid '.pdb'], 'file')
		            %try
		            PDBStruct = getpdb(ppi_str.pdbid, 'ToFile', ...
		                                ['/home/du/Protein_Protein_Interaction_Project/PDB/' ppi_str.pdbid '.pdb']);
		            %catch
		            %log_line = ...
		            %['PDB file not found: ' ppi_str.pdbid '.\n'];
		            %fprintf(logf_id, log_line);
		            %end
		        end
		        
		    else
		        
		        % read the following res-res interaction.
		        resres_row = textscan(cell_ddi{line}, '%s %s %s %s %s');
		        aaA = resres_row{1}{1};
		        aaB = resres_row{2}{1};
		        seqResA = resres_row{3}{1};
		        seqResB = resres_row{4}{1};
		        ppi_str.interactA(end+1) = {seqResA};
		        ppi_str.interactB(end+1) = {seqResB};
		        ppi_str.intaaA(end+1) = aaA;
		        ppi_str.intaaB(end+1) = aaB;
		        
		    end
		
		elseif strcmp(state, 'end-of-ddi')
		    
		    % we're done!
		    break;
		    
		end
	    
	    %{
	    catch exc
		fclose('all');
		error_msg = ['Problems at line ' num2str(line) '.\n'];
		error(error_msg);
	    end
	    %}
	end
end
return;