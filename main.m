%% AudioCLEF 2015 classification framework
%
% Written by Carmine E. Cella, ENS - Paris
% contact: carmine.emanuele.cella@ens.fr
%

close all;

rng(1);

addpath('lib');
if ~exist('libs_folder', 'var');
    libs_folder = '../../libs/';
end
addpath(genpath([libs_folder, 'scattering.m/']));
addpath([libs_folder, 'ScatNetLight/dimensionality_reduction/']);
addpath(genpath([libs_folder, 'ScatNetLight/svm_robust/']));
addpath([libs_folder, 'mfcc']);

%% parameters and structures
if ~exist('datasets_folder', 'var')
    datasets_folder = '../../datasets/';
end
%db_params = struct ('location', [datasets_folder, 'solosDb']);                % 20  classes
%db_params = struct ('location', [datasets_folder, 'bird_fake']);              % 2   classes
%db_params = struct ('location', [datasets_folder, 'minibird']);                % 15  classes
db_params = struct ('location', [datasets_folder, 'BD50CLASSES/FOLDERS']);    % 50  classes
%db_params = struct ('location', [datasets_folder, 'BirdCLEF_2014_folders']);  % 500 classes

features_params = struct ('type', 'mfcc', ...
    'mfcc_minf', 500, ...
    'mfcc_maxf', 16001, ....
    'mfcc_bands', 40, ...
    'mfcc_ceps', 13, ...
    'mfcc_win', .025, ...
    'mfcc_hop', .025, ...
    'scat_type', 'scat2', ...
    'scat_tw', 2048, ...
    'scat_Q', 16, ...
    'scat1_max_coeff', 90, ...
    'scat_norm', 'yes', ...
    'alogc_win', 2048, ...
    'alogc_olap', 512, ...
    'alogc_nbands', 40, ...
    'alogc_ncoeff', 13, ...
    'alogc_alpha', 3.5, ...
    'rms_norm', 'yes', ...
    'log_features', 'no', ...
    'thresholding', 'no', ...
    'crop_length', 250, ...
    'nCrops', 5, ...
    'detection_function', 'spectrum_energy', ...
    'feature_percentile', 0, ...
    'debug', 'no');

summarization_params = struct ('type', 'none', ...
    'components', 5);

equalization_params = struct ('type', 'none');

batch_params = struct ('Nfolds', 3, ...
    'tt_ratio', .25, ...
    'dimensions', 0, ...
    'standardize', 'no', ...
    'structured_validation', 'yes');

learning_params = struct ('type', 'none', ...
    'pca_whitening', 'no', ...
    'K', 40);

classification_params = struct ('type', 'none', ...
    'svm_kernel', 'rbf', ...
    'svm_sigma_v', 1.1, ...
    'svm_C', 10, ...
    'RF_ntree', 30, ...
    'RF_fboot', 1, ...  
    'histogram_voting', 'none', ...
    'debug', 'no');

torch_params = struct ('type', 'label', ...    
    'mode', 'frame_splitted', ...
    'tt_ratio', .25);

%% run classifcation
[F, labels, entries, acc, map, kernels] = AC_batch (db_params, features_params, learning_params, ...
    summarization_params, equalization_params, batch_params, classification_params);

%% export data to Torch
AC_make_torch_batch (F, labels, entries, torch_params, features_params.crop_length)

% eof
