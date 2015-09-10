function predictContactsTopologyLevelFromTopology_2010_LOO(ddi_folder)
%read topology list file

TopologyListFile=[ddi_folder '/topologyList.txt'];
fid = fopen(TopologyListFile, 'r');
topologyList = textscan(fid, '%s', 'delimiter', '\n');
topologyList = topologyList{1};
fclose(fid);

currDir = pwd;
%loop through topology level
for topology = 1:length(topologyList)
	
	topologyName=topologyList{topology};
	topology_folder=[ddi_folder '/' topologyName];
        ddiStructFile = [topology_folder '/ddi_str_array.mat'];
        load(ddiStructFile);

        [SimilarityVector, recallVector, specificityVector, precisionVector, accuracyVector, F1Vector, MCCVector, scoreVector] = ...
                        predictContactsFamilyLevel_singleDDI_2010_LOO(topology_folder, ddi_str_array);
        saveFile = [topology_folder '/SimilarityVector.txt'];
        save(saveFile,'SimilarityVector','-ascii');
      
      	saveFile = [topology_folder '/recallVector.txt'];
        save(saveFile,'recallVector','-ascii');

        saveFile = [topology_folder '/specificityVector.txt'];
        save(saveFile,'specificityVector','-ascii');
        
        saveFile = [topology_folder '/precisionVector.txt'];
        save(saveFile,'precisionVector','-ascii');
        
        saveFile = [topology_folder '/accuracyVector.txt'];
        save(saveFile,'accuracyVector','-ascii');
 
        saveFile = [topology_folder '/F1Vector.txt'];
        save(saveFile,'F1Vector','-ascii');
        
        saveFile = [topology_folder '/MCCVector.txt'];
        save(saveFile,'MCCVector','-ascii');
        
        saveFile = [topology_folder '/scoreVector.txt'];
        save(saveFile,'scoreVector','-ascii');                
        
        
end

return;
