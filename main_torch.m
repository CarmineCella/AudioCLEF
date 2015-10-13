%% Experimental module for Torch deep network classification

addpath ('lib')
addpath ('torch')

if ~exist('F', 'var');
   error ('AudioCLEF: you need to compute features at least once'); 
end


%% params

torch_tt_ratio = .5;
dimensions = 2;


[~, ~, F] = AC_standardization(F); % standardization is mandatory!
if (dimensions ~= 0)
    F_orig = F;
    [mu,E,V] = AC_pca (F');
    M = dimensions; 
    F = (V(:,1:M)')*(F-repmat(mu',[1 size(F,2)])); 
end

AC_make_torch_batch ('AC_torch_batch.h5', F, labels, entries, torch_tt_ratio)
disp('data written in H5 format; now evaluate nn_classification.lua in torch folder...')

% FIXME: this file will be merged in AC_classification.m

        