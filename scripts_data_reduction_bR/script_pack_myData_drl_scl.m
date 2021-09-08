% script_pack_myData_drl_scl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads Partialator parameters of crystFEL (OSF & relB) and apply them 
% to the snapshots, which are already passed the DRL removal step.
% A. Hosseini, Dec-2017, May-2018, July-2018, March-2019
% Cecilia C., modified 22 March 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

tic
%_________________________________________________________________________________________
% To load      
fileData_DRL = 'data_bR_dark_int_DRL_nS234988_nBrg54396.mat';         
%scalePara_file = 'light_sorted_collection_time_info.mat';
scalePara_file = 'sorted_info.mat';

% To save:
fileData_DRL_SCL = 'data_bR_dark_int_DRL_SCL_nS234988_nBrg54396.mat'; 

% lattice sizes for group P6(3):
a = 62.32; % in Angstrom
b = 62.32; % in Angstrom
c = 111.1; % in Angstrom

% load data: 
load(fileData_DRL,'T_drl','M_drl','miller_h','miller_k','miller_l'); 
load(scalePara_file, 'scales_dark');

scales = scales_dark;
clear scales_dark
OSF = scales(:,1);
relB = scales(:,2);

%_________________________________________________________________________________________
% Scaling with OSF and relB parameters:

qvec = [miller_h./a, miller_h./(sqrt(3)*a) + 2*miller_k./(sqrt(3)*b), miller_l./c];
q2 = qvec(:,1).^2 + qvec(:,2).^2 + qvec(:,3).^2; 
%notice: sinTheta_lambda2 = q2/4 = 1/(4*d2) 

T_drl_scl = sparse(size(T_drl,1),size(T_drl,2));
blocksize = 10000;  % for large matrix multiplication
N = blocksize;

nBrg = size(T_drl, 2)
for i = 1: ceil(size(T_drl,1)/N)
    i
  if i*N > size(T_drl,1)
    T_drl_scl(1+(i-1)*N:end,:) = T_drl(1+(i-1)*N:end,:) ...
                                 .* repmat(exp(-OSF(1+(i-1)*N:end)),1,nBrg) ...
                                 .* exp(relB(1+(i-1)*N:end) * q2'/4); 
  else
    T_drl_scl(1+(i-1)*N:i*N,:) = T_drl(1+(i-1)*N:i*N,:) ...
                                 .* repmat(exp(-OSF(1+(i-1)*N:i*N)),1,nBrg) ...
                                 .* exp(relB(1+(i-1)*N:i*N) * q2'/4); 
  end  
end

M_drl_scl = M_drl;

% save data with DRL cuts and scaled:
save(fileData_DRL_SCL,'T_drl_scl','M_drl_scl','miller_h','miller_k','miller_l','-v7.3');  

toc
%EOF