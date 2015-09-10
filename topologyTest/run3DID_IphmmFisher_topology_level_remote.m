function run3DID_IphmmFisher_topology_level_remote(file_ddis)

% logfile.
diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_IphmmFisher_' dateFormatted '.txt'];
if exist(logfile_name, 'file')
    command = ['rm -f ' logfile_name];
    system(command);
end
diary(logfile_name);

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' STARTED, IphmmFisher.\n'];
fprintf(logline);

% we'll only do the ddis in this dataset.
%file_ddis = 'ddisToFinishIphmmFisher_10MAY2011.txt';
fid = fopen(file_ddis, 'r');
cell_ddis = textscan(fid, '%s', 'delimiter', '\n');
cell_ddis = cell_ddis{1};
fclose(fid);

folder3did = ['/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/'];

currDir = pwd;
for ddi_ctr = 1:length(cell_ddis)
    
    cd(currDir);
    
    %{
    ddi_line = textscan(cell_ddis{ddi_ctr}, '%s %s');
    domAPfam = ddi_line{1}{1};
    domBPfam = ddi_line{2}{1};
    ddiName = [domAPfam '_int_' domBPfam];
    %}
    ddiName = cell_ddis{ddi_ctr};
    %topologyList = The topolgy list in the topolgyListFile;
    folder_ddi = [folder3did ddiName '/'];
    TopologyListFile=[folder_ddi 'topologyList.txt'];
    fid = fopen(TopologyListFile, 'r');
    topologyList = textscan(fid, '%s', 'delimiter', '\n');
    topologyList = topologyList{1};
    fclose(fid);
    for topology = 1:length(topologyList)
	    topologyFolder=[folder_ddi topologyList{topology} ''];

	    ddiStructFile = [topologyFolder '/ddi_str_array.mat'];
	    try
		% load the ddi structure.
		%ddiStructFile = [folder3did ddiName '/ddi_str_array.mat'];
		load(ddiStructFile);
		[iphmmA iphmmB ...
		    AFisherM0Array AFisherM1Array ...
		    AconstFisherM0Array AconstFisherM1Array ...
		    MSA_domA ConsSeqA int_sitesA ...
		    BFisherM0Array BFisherM1Array ...
		    BconstFisherM0Array BconstFisherM1Array ...
		    MSA_domB ConsSeqB int_sitesB] = ...
		                runDDIFamily_IphmmFisher_topology_level_remote(ddi_str_array);
		saveFile = [topologyFolder '/iphmmA.mat'];
		save(saveFile, 'iphmmA');
		saveFile = [topologyFolder '/iphmmB.mat'];
		save(saveFile, 'iphmmB');
		saveFile = [topologyFolder '/FisherA.mat'];
		save(saveFile, 'AFisherM0Array', 'AFisherM1Array', ...
		                'AconstFisherM0Array', 'AconstFisherM1Array');
		saveFile = [topologyFolder '/FisherB.mat'];
		save(saveFile, 'BFisherM0Array', 'BFisherM1Array', ...
		                'BconstFisherM0Array', 'BconstFisherM1Array');
		saveFile = [topologyFolder '/MSA_domA.mat'];
		save(saveFile, 'MSA_domA');
		try
		saveFile = [topologyFolder '/MSA_domA.txt'];
		multialignwrite(saveFile, MSA_domA);
		catch
		    % sometimes the MSA can't be written because the family only
		    % has one example.
		    warning('MSA A could not be written');
		end
		saveFile = [topologyFolder '/MSA_domB.mat'];
		save(saveFile, 'MSA_domB');
		try
		saveFile = [topologyFolder '/MSA_domB.txt'];
		multialignwrite(saveFile, MSA_domB);
		catch
		    % sometimes the MSA can't be written because the family only
		    % has one example.
		    warning('MSA B could not be written');
		end
		saveFile = [topologyFolder '/ConsSeqA.mat'];
		save(saveFile, 'ConsSeqA');
		saveFile = [topologyFolder '/ConsSeqB.mat'];
		save(saveFile, 'ConsSeqB');
		saveFile = [topologyFolder '/int_sitesA.mat'];
		save(saveFile, 'int_sitesA');
		saveFile = [topologyFolder '/int_sitesB.mat'];
		save(saveFile, 'int_sitesB');
		
		% finished! report SUCCESS to log file.
		c = clock;
		time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
		        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
		        num2str(c(5), '%0.0d')];
		logline = ...
		['\n' time ' SUCCESS: ' ddiName ', IphmmFisher.\n'];
		fprintf(logline);
		
	    catch exc   % block that trains the two iphmms.
		cd(currDir);
		c = clock;
		time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
		        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
		        num2str(c(5), '%0.0d')];
		logline = ['\n' time ' ERROR: ' ddiName ...
		                    ', IphmmFisher, Message: ' exc.message '\n'];
		fprintf(logline);
		% print the stack.
		for stackCtr = length(exc.stack):-1:1
		    fprintf([exc.stack(stackCtr).file '; ' ...
		                exc.stack(stackCtr).name '; ' ...
		                num2str(exc.stack(stackCtr).line) '\n']);
		end
	    end
    end
end

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' FINISHED, IphmmFisher.\n'];
fprintf(logline);

diary off;

return;
