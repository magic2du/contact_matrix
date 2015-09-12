function savetsv(matrixFilePath, score_contactMatrix)

    fid = fopen(matrixFilePath,'w');
    %firstLine='SeqA' SeqB\tScore\n';
    fprintf(fid,'%s\t%s\t%s\n','SeqA', 'SeqB', 'Score');    
%    [aLength,bLength]=size(score_contactMatrix)
    for i=1:aLength
     for j=1:bLength

        fprintf(fid,'%s\t%s\t%s\n',num2str(i), num2str(j), num2str(score_contactMatrix(i,j)));
     end
    end
    fclose(fid);
    
return
