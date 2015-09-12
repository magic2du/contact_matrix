function value=hydropathy_index(AA)
% function that return the hydropathy index of an amino acid.
hydropathy.('A')=1.8;
hydropathy.('R')=-4.5;
hydropathy.('N')=-3.5;
hydropathy.('D')=-3.5;
hydropathy.('C')=2.5;
hydropathy.('E')=-3.5;
hydropathy.('Q')=-3.5;
hydropathy.('G')=-0.4;
hydropathy.('H')=-3.2;
hydropathy.('I')=4.5;
hydropathy.('L')=3.8;
hydropathy.('K')=-3.9;
hydropathy.('M')=1.9;
hydropathy.('F')=2.8;
hydropathy.('P')=-1.6;
hydropathy.('S')=-0.8;
hydropathy.('T')=-0.7;
hydropathy.('W')=-0.9;
hydropathy.('Y')=-1.3;
hydropathy.('V')=4.2;
value=hydropathy.(upper(AA));

	

end
