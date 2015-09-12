function run3DID_STAT_200_500_multipTop_examples_2010()
%collect the training exmples and validation(last sequence pair) 
%feature vectors{F0=20,F1=20, Sliding=17*11}*2 and output={0,1}. output 1 means residuepairs are contact pairs. lenght=455
%For each sequence, build a matrix index by the sequence number. I decided to write it to different DDI direcoty instead of just one big file. No longer write training and validation set.
%outputPath=['/big/du/Protein_Protein_Interaction_Project/Contact_Matrix_Project/Vectors_Fishers_aaIndex_raw/' ddiddiName '/']
file_ddis='listOfDDIsHaveOver2InterfacesHave200-1000Examples_pseudo.txt';
diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_run3DID_ STAT_200_500_examples_2010' ...
             dateFormatted '.txt'];
if exist(logfile_name, 'file')
    command = ['rm -f ' logfile_name];
    system(command);
end
diary(logfile_name);

c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' STARTED.\n'];
fprintf(logline);


%file_ddis = ...
%'finishedDDIs_AllVectorsChooseNegRand_FisherM1_SVMLIGHT_POLY_02NOV2011.txt';
fid = fopen(file_ddis, 'r');
cell_ddis = textscan(fid, '%s', 'delimiter', '\n');
cell_ddis = cell_ddis{1};
fclose(fid);

currDir = pwd;
successCtr = 0;
ddisWith_200to250Examples=[];

%%%%%% parallel starts%%%%%%%%%%
for ddi_ctr = 1:length(cell_ddis)
    cd(currDir);

    ddiName = cell_ddis{ddi_ctr};
    
    try
        
	
        ddiStructFile=['/home/du/Protein_Protein_Interaction_Project/' ...
                                        '3did_15OCT2010/dom_dom_ints/' ddiName '/ddi_str_array.mat'];

	load(ddiStructFile);
	num=length(ddi_str_array);
	num
	if (num>=200 && num <= 500)
		ddisWith_200to250Examples{end+1}=ddiName;
	else
		continue
	end
        
        
        % finished! report SUCCESS to log file.
        successCtr = successCtr + 1;
        c = clock;
        time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
                num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
                num2str(c(5), '%0.0d')];
        logline = ['\n\n\n' time ' SUCCESS: ' ddiName ...
                    ];
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

save('STAT_200_500_examples_2010_real.txt', 'ddisWith_200to250Examples', '-ascii' )
cd(currDir);
c = clock;
time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
        num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
        num2str(c(5), '%0.0d')];
logline = [time ' FINISHED.\n'];
fprintf(logline);

diary off;


return;
