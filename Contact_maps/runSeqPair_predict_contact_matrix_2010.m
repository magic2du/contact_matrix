function contactMatrix = runSeqPair_predict_contact_matrix_2010 ...
                        (ddiName, SeqA, SeqB, tsvPath)
% get the contract score Matrix given squences and ddiname                         
%%%%%%%%%%%%%%testing%%%%%%%%%%%%%%%%%%%%%%%
%function auc = runSeqPairAllVectorsChooseNegRand_02NOV2011()
%{
pairNbr = 17;
ddiName = 'PF00385.16_int_PF00385.16';

%}
tsvPath= '/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/';
ddiName = 'Homoserine_dh_int_NAD_binding_3';
SeqA='PIISFLREIIQTGDEVEKIEGIFSGTLSYIFNEFSTSQANDVKFSDVVKVAKKLGYTEPDPRDDLNGLDVARKVTIVGRISGVEVESPTSFPVQSLIPKPLESVKSADEFLEKLSDYDKDLTQLKKEAATENKVLRFIGKVDVATKSVSVGIEKYDYSHPFASLKGSDNVISIKTKRYTNPVVIQGAGAGAAVTAAGVLGDVIGSNVEFMCKVYSDPQPHIQWLKHVQILKTAGVNTTDKEMEVLHLRNVSFEDAGEYTCLA';
 
SeqB='GAGVVGSAFLDQLLAMKSTITYNLVLLAEAERSLISKDFSPLNVGSDWKAALAASTTKTLPLDDLIAHLKTSPKPVILVDNTSSAYIAGFYTKFVENGISIATPNKKAFSSDLATWKALFSNKPTNGFVYHEDPKRLYCKNGGFFLRIHPDGRVDGVREKSDPHIKLQLQAEERGVVSIKGVSANRYLAMKEDGRLLASKSVTDECFFFERLESNNYNTYRSRKYTSWYVALKRTGQYKLGSKTGPGQKAILFL';
%%%%%%%%%%%%%%log%%%%%%%%%%%%%%%%%%%%%%%
diary off;
dateFormatted = upper(date);
dateFormatted = dateFormatted(regexp(upper(date), '[A-Z0-9]'));
logfile_name = ['log_runSeqPair_predict_contact_matrix_2011_' dateFormatted '.txt'];
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
%%%%%%%%%%%%%%log end%%%%%%%%%%%%%%%%%%%
% set up parameters%
FisherMode = 'FisherM1';
SVMMode = 'SVMLIGHT';
Kernel = 'POLY';
folder3did = ['/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/'];
ddiPath = [folder3did ddiName '/'];
SVMmodelFile = [ddiPath FisherMode '_'  SVMMode '_' Kernel '_' 'build_matrix_model_with_contactMaps.model']
% main part%
try

    if ~ exist(SVMmodelFile, 'file') % if the SVM model does not exist for this DDI
        'no model file'
        runSeqPair_build_matrix_model(ddiName, FisherMode, SVMMode, Kernel);
    else
        contactMatrix = predict_contact_matrix_with_SVMmodel(ddiName, SeqA, SeqB, FisherMode, SVMMode, Kernel, SVMmodelFile, tsvPath);
    end
    c = clock;
    time = [num2str(c(1), '%0.0d') '-' num2str(c(2), '%0.0d') '-' ...
            num2str(c(3), '%0.0d') ' ' num2str(c(4), '%0.0d') ':' ...
            num2str(c(5), '%0.0d')];
    logline = ['\n\n\n' time ' SUCCESS: ' ddiName ...
                '.\n\n\n'];
    fprintf(logline);
catch exc

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
return;
