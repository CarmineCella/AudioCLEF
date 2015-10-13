%% Experimental module for Torch deep network classification

addpath ('torch')

if ~exist('F', 'var');
   error ('AudioCLEF: you need to compute features at least once'); 
end


%% params

torch_tt_ratio = .25;

[~, ~, F] = AC_standardization(F); % standardization is mandatory!
AC_make_torch_batch ('AC_torch_batch.h5', F, labels, entries, torch_tt_ratio)
disp('data written in H5 format; now evaluate nn_classification.lua in torch folder...')

        