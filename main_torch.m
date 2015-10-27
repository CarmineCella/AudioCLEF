%% Experimental module for Torch deep network classification

addpath ('lib')
addpath ('torch')

if ~exist('F', 'var');
   error ('AudioCLEF: you need to compute features at least once'); 
end


%% params

torch_params = struct ('type', 'probability', ...
    'mode', 'monolithic', ...
    'tt_ratio', .3, ...
    'dimensions', 0);

AC_make_torch_batch (F, labels, entries, torch_params)
disp('data written in H5 format; now evaluate nn_classification.lua in torch folder...')

% FIXME: this file will be merged in AC_classification.m

       