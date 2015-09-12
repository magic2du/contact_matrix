function runSeqPair_build_matrix_model_2011(ddiName, FisherMode, SVMMode, Kernel)
    
    
    % check if the contactMapsBag exists    
    contactMatrixPath = ...
            ['/home/du/Protein_Protein_Interaction_Project/3did_Apr2011/dom_dom_ints/' ...
                                            ddiName '/contactMapsBag.mat'];
                            
    if ~ exist(contactMatrixPath, 'file') % if the SVM model doesn't exist for this DDI
        build_contactMaps_2011(ddiName);
	build_matrix_model_with_contactMaps_2011(ddiName, FisherMode, SVMMode, Kernel);
        
    else
        build_matrix_model_with_contactMaps_2011(ddiName, FisherMode, SVMMode, Kernel);
    end

end
