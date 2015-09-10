function run3DID_trainDDI_SVDSVM_FisherM0_Light(file_ddis)

diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
%logfile_name = 'log_trainDDI_SVDSVM_FisherM0_Toy_21APR2011.txt';
logfile_name = ['log_trainDDI_SVDSVM_FisherM0_' dateFormatted '.txt'];
if exist(logfile_name, 'file')
%command = ['rm -f ' logfile_name];
% system(command);
	logfile_name = [logfile_name '1.txt'];
end
diary(logfile_name);

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' STARTED.\n'];
fprintf(logline);

% we'll only do the ddis in this dataset.
%file_ddis = 'toyDataset.txt';

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
    ddi_folder = [domAPfam '_int_' domBPfam];
    %}
    ddi_folder = cell_ddis{ddi_ctr};
    
    try
        trainDDI_SVDSVM([folder3did ddi_folder], 'FisherM0', 'SVMLIGHT');
        
        % finished! report SUCCESS to log file.
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = ['\n\n\n' time ' SUCCESS: ' ddi_folder '.\n\n\n'];
        fprintf(logline);
        
    catch exc
        cd(currDir);
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = [time ' ERROR: ' ddi_folder ...
                                        '. Message: ' exc.message '\n'];
        fprintf(logline);
        % print the stack.
        for stackCtr = length(exc.stack):-1:1
            fprintf([exc.stack(stackCtr).file '; ' ...
                        exc.stack(stackCtr).name '; ' ...
                        num2str(exc.stack(stackCtr).line) '\n']);
        end
    end
    
end

cd(currDir);
c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' FINISHED.\n'];
fprintf(logline);

diary off;

return;
