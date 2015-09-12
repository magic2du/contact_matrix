import Bio
import Bio.PDB
import numpy
pdb_code="aIIb"
pdb_filename="alpha_IIB_2VDK_A_CIB1_2012-01-19.5185462227.pdb"
def calc_residue_dist(residue_one, residue_two) :
   """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return numpy.sqrt(numpy.sum(diff_vector * diff_vector))

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
     return numpy.sqrt(numpy.sum(diff_vector * diff_vector))


structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
model=structure[0]
dist_matrix = calc_dist_matrix(model["A"], model["B"])
contact_map = dist_matrix < 12.0
