function prettyPrintMatrix(matrix, integer, filename, ...
                                        horizHeader, vertHeader, separator)

if ~isempty(filename)
    fid = fopen(filename, 'w');
end

% if no headers provided.
if isempty(horizHeader)
    for rows = 1:size(matrix, 1)
        line = '';
        for cols = 1:size(matrix, 2)
            if integer
                printNbr = num2str(matrix(rows, cols));
            else
                printNbr = num2str(matrix(rows, cols), '%0.2f');
            end

            % mind pretty tabbing. 
            if length(printNbr) >= 4 
                line = [line printNbr separator];
            else
                line = [line printNbr separator separator];
            end
        end
        line = [line '\n'];
        if ~isempty(filename)
            fprintf(fid, line);
        else
            fprintf(line);
        end
    end

% headers provided.
else
    % first row.
    line = [separator separator];
    for cols = 1:size(matrix, 2)
        line = [line horizHeader(cols) separator separator];
    end
    line = [line '\n'];
    if ~isempty(filename)
        fprintf(fid, line);
    else
        fprintf(line);
    end
    
    % all other rows.
    for rows = 1:size(matrix, 1)
        line = [vertHeader(rows) separator separator];
        for cols = 1:size(matrix, 2)
            if integer
                %try
                printNbr = num2str(matrix(rows, cols));
                %catch
                %    stop = 1;
                %end
            else
                printNbr = num2str(matrix(rows, cols), '%0.2f');
            end

            % mind pretty tabbing. 
            if length(printNbr) >= 4 
                line = [line printNbr separator];
            else
                line = [line printNbr separator separator];
            end
        end
        line = [line '\n'];
        if ~isempty(filename)
            fprintf(fid, line);
        else
            fprintf(line);
        end
    end
end

if ~isempty(filename)
    fclose(fid);
end
                                            
return;