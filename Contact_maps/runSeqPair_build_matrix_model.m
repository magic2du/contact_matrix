function runSeqPair_build_matrix_model(ddiName, FisherMode, SVMMode, Kernel)
    
    
    % check if the contactMapsBag exists    
    contactMatrixPath = ...
            ['/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/dom_dom_ints/' ...
                                            ddiName '/contactMapsBag.mat'];
                            
    if ~ exist(contactMatrixPath, 'file') % if the SVM model exist for this DDI
        build_contactMaps(ddiName);
        
    else
        build_matrix_model_with_contactMaps(ddiName, FisherMode, SVMMode, Kernel);
    end

end
