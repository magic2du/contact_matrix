function write_SVM_file(PO, NE, file_name)

fid = fopen(file_name, 'w');
% write positive examples.
for rows = 1:size(PO, 1)
    line = '1 ';
    for cols = 1:size(PO, 2)
        line = ...
        [line num2str(cols, '%d') ':' num2str(PO(rows, cols), '%f') ' '];
    end
    fprintf(fid, [line '\n']);
end

% write negative examples.
for rows = 1:size(NE, 1)
    line = '-1 ';
    for cols = 1:size(NE, 2)
        line = ...
        [line num2str(cols, '%d') ':' num2str(NE(rows, cols), '%f') ' '];
    end
    fprintf(fid, [line '\n']);
end

fclose(fid);

return;