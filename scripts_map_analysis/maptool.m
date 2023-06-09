% maptool_bR
% -------------------------------------------------------------------------
% written by Cecilia Wickstrand, 05-04-2020
% modified by Cecilia Casadei, 2021
%
% publication: 
% "A tool for visualizing protein motions in time-resolved crystallography"
% published online 01-04-2020, in Structural Dynamics (Vol.7, Issue 2).
% https://doi.org/10.1063/1.5126921 
%
% For analysis of a set of difference Fourier electron density maps from 
% time-resolved serial femtosecond crystallography (TR-SFX) experiments on
% bacteriorhodopsin. 
% - Difference electron density within a sphere around every atom in 
%   the resting state structure is extracted 
% - The average positive and negative density over the sphere is
%   calculated for each map, with or without a sigma cut-off
% - The resulting dual amplitude function may be plotted along the trace of
%   selected atoms or used for further analysis
%
% This script is an example for analysis of three maps (16 ns, 760 ns and 1.725 ms).
% Before analysis, execute "process_maps.sh" to convert the maps to 
% cartesian coordinates and .h5 format.
% -------------------------------------------------------------------------

% INPUT
% -------------------------------------------------------------------------
clear all;

% Sphere and sigma settings
radius = 1.7; % ??
distance = 0.2; % ??, how dense grid within sphere
%sigmacutoff = 4; % exclude data below this sigma level



%label = '_step_10_range_178000_195000';
% Files
here = '.';
pdbpath = [here '/6g7k_C20.pdb']; % resting state pdb
indir  = [here '/output/']; % where to find .h5 maps
outdir = [here '/results/'];

nrmaps = 1046;
mapnames = cell(nrmaps, 1);
for idx = 1:nrmaps
    i = 100*(idx-1);
    nm = ['1.8_bR_light_p_0_1_modes_timestep_' num2str(i, '%0.6d') '_light--dark_I_dark_avg'];
    mapnames(idx,1) = {nm};
end


% START CALCULATIONS
% ------------------------------------------------------------------------
time = tic;

% LOAD ATOMS
pdb = pdbread(pdbpath);

% In this case all HETATM are written as ATOM in pdb
atomcoord = [[pdb.Model.Atom.X]' [pdb.Model.Atom.Y]' [pdb.Model.Atom.Z]'];
nratoms =  size(atomcoord,1); 

% CALCULATE SPHERE COORDINATES
calcdist = @(x,y,z) sqrt(x^2 + y^2 + z^2);
 
% radius 2 distance 0.5 -> spots = [-2 -1.5 -1 -0.5 0 0.5 1 1.5 2]
spots = distance:distance:ceil(radius/distance)*distance;
spots = [-fliplr(spots) 0 spots];
nrspots = length(spots);
 
% spherelist column 1 2 3 = coordinates
spherelist= zeros(nrspots^3,7);
count = 0;
for i = 1:nrspots
   for j = 1:nrspots
        for k = 1:nrspots
            dist = calcdist(spots(i),spots(j),spots(k));
            if dist <= radius
               count = count + 1;
               spherelist(count,:) = [spots(i) spots(j) spots(k) i j k dist];
            end
        end
   end
end
spherelist = spherelist(1:count,:); 
nrpoints = size(spherelist,1);


% PRECALCULATE ALL COORDINATES IN ALL SPHERES
% (example: 1834 rows = atoms, 2109 columns = points) 
X = repmat(atomcoord(:,1),1,nrpoints)+repmat(spherelist(:,1)',nratoms,1);
Y = repmat(atomcoord(:,2),1,nrpoints)+repmat(spherelist(:,2)',nratoms,1);
Z = repmat(atomcoord(:,3),1,nrpoints)+repmat(spherelist(:,3)',nratoms,1);

% Reshape to a single column starting with first point for each atom, then second etc. 
X=reshape(X, nratoms*nrpoints,1);
Y=reshape(Y, nratoms*nrpoints,1);
Z=reshape(Z, nratoms*nrpoints,1);


% LOAD GRID
% (using first experimental XYZinfo file - here all files have the same grid)
XYZinfo = dlmread([indir 'log/' mapnames{1} '_XYZinfo.dat']);

celldimensions = XYZinfo(1,1:3);
gridpoints = XYZinfo(2,1:3); 
axislimits = XYZinfo(3,:);  
axisorder = XYZinfo(4,1:3);  % Used to extract correct axislimits for each axis

% grid distance
dX = celldimensions(1)/gridpoints(1);
dY = celldimensions(2)/gridpoints(2);
dZ = celldimensions(3)/gridpoints(3);

tmp = [axisorder(1) axislimits(1:2); axisorder(2) axislimits(3:4); axisorder(3) axislimits(5:6)];
axes = sortrows(tmp); % By def., in order 1 2 3 (ie x, y, z)

% points at each side
sX = [axes(1,2):axes(1,3)]*dX;
sY = [axes(2,2):axes(2,3)]*dY;
sZ = [axes(3,2):axes(3,3)]*dZ;

[gY,gZ,gX] = meshgrid(sY,sZ,sX); % order so that gY, gZ, gX have same dim as map.


% LOAD MAPS AND EXTRACT DENSITIES
% ------------------------------------------------------------------------
%nrmaps = length(mapnames);
sigma = zeros(nrmaps, 2);
mapd0 = zeros(nrmaps, nratoms, nrpoints);

for m = 1:nrmaps
    fprintf(['Currently at map ' num2str(m) ' time ' num2str(toc(time)) ' s\n'])

    % Load sigma info
    XYZinfo = dlmread([indir 'log/' mapnames{m} '_XYZinfo.dat']);
    sigma(m,1:2) = XYZinfo(5:6,1);

    
    % LOAD MAP AND CALCULATE DENSITY AT POINTS BY INTERPOLATION
    % reshape back to rows = atoms, columns = points
    map = hdf5read([indir mapnames{m} '_cartesian.h5'],'map');
    tmp = interp3(gY,gZ,gX,map,Y,Z,X);

    mapdensities = reshape(tmp, nratoms, nrpoints);
    % divide with sigma of original map
    mapd0(m,:,:) = mapdensities/sigma(m,2); 
end

save([outdir 'mapd0.mat'], 'mapd0', '-v7.3');

% % CALCULATE AVERAGE DENSITIES 
% %------------------------------------
% % Calculate mean positive and negative densities
% mapd = mapd0;
% mapd(abs(mapd0) < sigmacutoff) = 0;
% clear 'mapd0';
% 
% mapd_pos = mapd;
% mapd_pos(mapd < 0) = 0;
% meanposden = mean(mapd_pos,3);
% clear 'mapd_pos';
% save([outdir '/meanposden_map_modes_0_', num2str(mode), '_radius_', num2str(radius), '_dist_', num2str(distance), '_sig_', num2str(sigmacutoff), '.mat'], 'meanposden');
% clear 'meanposden';
% 
% mapd_neg = mapd;
% mapd_neg(mapd > 0) = 0;
% meannegden = mean(mapd_neg,3);
% clear 'mapd_neg';
% save([outdir '/meannegden_map_modes_0_', num2str(mode), '_radius_', num2str(radius), '_dist_', num2str(distance), '_sig_', num2str(sigmacutoff), '.mat'], 'meannegden');
% clear 'meannegden';
% 
% save([here '/bov_nlsa_refine_96_edited.mat'], 'pdb');


     
% OPTIONAL CONTROL SECTION
%**************************************************************************
% - CHECK GRID
%   load a map
%   map1 = hdf5read([indir mapnames{1} '_cartesian.h5'],'map');
%   check that the size of the map is the same as for gX / gY / gZ 
%   if not, change the order of Y Z X in the meshgrid command 
% 
% - CHECK PDB IS COVERED BY MAP
%   pdb = pdbread(pdbpath);
%   lim_map = [sX(1) sX(end) sY(1) sY(end) sZ(1) sZ(end)];
%   lim_pdb = [min([pdb.Model.Atom(:).X]) max([pdb.Model.Atom(:).X]) min([pdb.Model.Atom(:).Y]) max([pdb.Model.Atom(:).Y]) min([pdb.Model.Atom(:).Z]) max([pdb.Model.Atom(:).Z])];
% 
% - CHECK DENSITY CALCULATIONS
%   Select a map and a pick few random atoms
%   Calculate the interpolated density value at each of the atoms
%   Open coot, go to the atoms and set the contour level to see that the
%   calculated densities are correct.
% 
%   testset = [157;307;457;1657]
%   testpdb.Model.Atom = pdb.Model.Atom(testset)
%   aX = [testpdb.Model.Atom.X]';
%   aY = [testpdb.Model.Atom.Y]';
%   aZ = [testpdb.Model.Atom.Z]';
%   map3 = hdf5read([indir mapnames{3} '_cartesian.h5'],'map');
%   testdensities = interp3(gY,gZ,gX,map3, aY, aZ , aX)
%  
%   For this test set (THR 24 C; TYR 43 OH; GLY 63 C; VAL 217 O)
%   the calculated densites (-0.0269    0.0157    0.0657   -0.0954)
%   are identical to those seen in coot.
%**************************************************************************