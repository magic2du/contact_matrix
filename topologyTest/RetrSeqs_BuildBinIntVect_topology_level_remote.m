function RetrSeqs_BuildBinIntVect_topology_level_remote(file_ddis)

% go to ddi_struct arrays in input file, and fill in the seq and 
% interaction (binary) vectors.

% logfile.
diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_RetrSeqs_BuildBinIntVect_' dateFormatted '.txt'];
if exist(logfile_name, 'file')
    command = ['rm -f ' logfile_name];
    system(command);
end
diary(logfile_name);

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' STARTED, RetrSeqs_BuildBinIntVect.\n'];
fprintf(logline);

%file_ddis = 'ddisFinishedDomainIphmmFisher_19NOV2011.txt';
fid = fopen(file_ddis, 'r');
cell_ddis = textscan(fid, '%s', 'delimiter', '\n');
cell_ddis = cell_ddis{1};
fclose(fid);

currDir = pwd;
successCtr = 0;

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
        ddi_seqs_and_intvectors_topology_level_remote(ddiName);
        
        % finished! report SUCCESS to log file.
        successCtr = successCtr + 1;
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = ['\n\n\n' time ' SUCCESS: ' ddiName ...
                                ', RetrSeqs_BuildBinIntVect.\n\n\n'];
        fprintf(logline);
        
        %{
        if successCtr == 20
            % force end.
            break;
        end
        %}
        
    catch exc
        cd(currDir);
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = [time ' ERROR: ' ddiName ...
            ', RetrSeqs_BuildBinIntVect, Message: ' exc.message '\n'];
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
logline = [time ' FINISHED, RetrSeqs_BuildBinIntVect.\n'];
fprintf(logline);

diary off;

return;
