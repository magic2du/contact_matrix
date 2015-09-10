function calculateIPHMM(domA, domB)

%domA = 'Homoserine_dh';
%domB = 'NAD_binding_3';

pfamDir = ...
'/home/du/Protein_Protein_Interaction_Project/PFAM_2008/SINGLE_FILES/';
phmmA = pfamhmmread([pfamDir domA '.pfam']);
phmmB = pfamhmmread([pfamDir domB '.pfam']);
dir3did = ...
'/home/du/Protein_Protein_Interaction_Project/3did_15OCT2010/topologyTest/dom_dom_ints/';
ddiDir = [dir3did domA '_int_' domB '/'];
load([ddiDir 'ddi_str_array.mat']);
[iphmmA, iphmmB] = generate_iphmm(ddi_str_array, phmmA, phmmB);

save([ddiDir 'iphmmA.mat'], 'iphmmA');
save([ddiDir 'iphmmB.mat'], 'iphmmB');

return;
