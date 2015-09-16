function predictContactsFamilyLevelFromList_2010(file_ddis)

% logfile.
diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_predictContactsFamilyLevelFromList_' dateFormatted '.txt'];
if exist(logfile_name, 'file')
    command = ['rm -f ' logfile_name];
    system(command);
end
diary(logfile_name);

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' STARTED, predictContactsFamilyLevelFromList.\n'];
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
    
    try
        % load the ddi structure.
        ddiStructFile = [folder3did ddiName '/ddi_str_array.mat'];
        load(ddiStructFile);
        ddi_folder = [folder3did ddiName ''];
        [SimilarityVector, recallVector, specificityVector, precisionVector, accuracyVector, F1Vector, MCCVector, scoreVector] = ...
                        predictContactsFamilyLevel_singleDDI_2010(ddi_folder, ddi_str_array);
        saveFile = [folder3did ddiName '/SimilarityVector.txt'];
        save(saveFile,'SimilarityVector','-ascii');
        
       	saveFile = [folder3did ddiName '/recallVector.txt'];
        save(saveFile,'recallVector','-ascii');

        saveFile = [folder3did ddiName '/specificityVector.txt'];
        save(saveFile,'specificityVector','-ascii');
        
        saveFile = [folder3did ddiName '/precisionVector.txt'];
        save(saveFile,'precisionVector','-ascii');
        
        saveFile = [folder3did ddiName '/accuracyVector.txt'];
        save(saveFile,'accuracyVector','-ascii');
 
        saveFile = [folder3did ddiName '/F1Vector.txt'];
        save(saveFile,'F1Vector','-ascii');
        
        saveFile = [folder3did ddiName '/MCCVector.txt'];
        save(saveFile,'MCCVector','-ascii');
        
        saveFile = [folder3did ddiName '/scoreVector.txt'];
        save(saveFile,'scoreVector','-ascii');        
        
        
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
                            ', predictContactsFamilyLevelFromList, Message: ' exc.message '\n'];
        fprintf(logline);
        % print the stack.
        for stackCtr = length(exc.stack):-1:1
            fprintf([exc.stack(stackCtr).file '; ' ...
                        exc.stack(stackCtr).name '; ' ...
                        num2str(exc.stack(stackCtr).line) '\n']);
        end
    end
    
end

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' FINISHED, predictContactsFamilyLevelFromList.\n'];
fprintf(logline);

diary off;

return;
