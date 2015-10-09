%% Experimental module for Torch deep network classification

addpath ('torch')

if ~exist('F', 'var');
   error ('AudioCLEF: you need to compute features at least once'); 
end


%% params

torch_tt_ratio = .1;

AC_make_torch_batch ('AC_torch_batch.h5', F, labels, entries, torch_tt_ratio)
% system ('~/torch/install/bin/th torch/data_loading.lua')
% system ('~/torch/install/bin/th torch/nn_classification.lua')

    