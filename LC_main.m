%% LifeClef 2015 classification framework
%
% Written by Carmine E. Cella, ENS - Paris
% contact: carmine.emanuele.cella@ens.fr
%

%clear all % Warning!!
close all

rng (1);

addpath(genpath('../../libs/scattering.m/'));
addpath('../../libs/ScatNetLight/dimensionality_reduction/');
%addpath('../../libs/libsvm_light'); % Mia Xu Chen
addpath(genpath('../../libs/ScatNetLight/svm_robust/'));
%addpath('../../libs/ScatNetLight/svm_robust/libsvm-compact-0.1/matlab/');

addpath ('../../libs/mfcc');
addpath ('../../libs/GMM');
%addpath(genpath('../../libs/randomforest-matlab/'));
addpath(genpath('../../libs/SPKmeans'));

addpath ('../HSC');

%% parameters and structures
%db_params = struct ('location', '../../datasets/Instrument_samples');            % 94 classes
%db_params = struct ('location', '../../datasets/Mogees_april_2015');            % 6 classes

%db_params = struct ('location', '../../datasets/bird_fake');                   % 2 classes
db_params = struct ('location', '../../datasets/minibird');                    % 15 classes
%db_params = struct ('location', '../../datasets/BD50CLASSES/FOLDERS');         % 50 classes
%db_params = struct ('location', '../../datasets/BirdCLEF_2014_folders');       % 500 classes

features_params = struct ('type', 'mfcc', ...
                          'mfcc_minf', 500, ...
                          'mfcc_maxf', 21000, ....
                          'mfcc_bands', 40, ...
                          'mfcc_ceps', 13, ...
                          'mfcc_win', .025, ... 
                          'mfcc_hop', .025, ...
                          'scat_type', 'scat1', ...
                          'scat_chunksize', 16384, ...
                          'scat_tw1', 2048, ...
                          'scat_tw2', 2048, ...
                          'scat_Q', 16, ...
                          'scat1_max_coeff', 90, ...
                          'rms_norm', 'yes', ...
                          'log_features', 'no', ...     
                          'thresholding', 'no', ...
                          'debug', 'no');        
                      
learning_params = struct ('type', 'none', ...
                          'pca_whitening', 'no', ...
                          'non_lin', 'module', ...
                          'K', 140);
                      
summarization_params = struct ('type', 'mean_std', ...
                               'components', 2); % FIXME: must be set accordingly!
                                            
classification_params = struct ('type', 'RF', ...
                                'svm_kernel', 'rbf', ...
                                'svm_sigma_v', .81, ...
                                'svm_C', .1, ...
                                'RF_ntree', 20, ...
                                'RF_fboot', 1, ...
                                'histogram_voting', 'no');           

%%% global parameters
equalize_distribution = 'no';
Nfolds = 3;
tt_ratio = .25;
dimensions = 0;
standardize = 'yes';    
structured_validation = 'yes';

saved_features = ''; % to decide if features must be recomputed
saved_db = '';
all_p = sprintf ('db_params:\n%s\nfeatures_params:\n%s\nlearning_params:\n%s\nsummarization_params:\n%s\nclassification_params:\n%s\n', ...
    evalc (['disp (db_params)']), evalc (['disp (features_params)']), evalc (['disp (learning_params)']), ...
    evalc (['disp (summarization_params)']), evalc (['disp (classification_params)']));              
                    
%% compute features
tstart = tic;
if exist('LC_features_and_labels.mat', 'file')  == 2
   load('LC_features_and_labels.mat');
end

if (strcmp (saved_features, evalc (['disp (features_params)'])) ~= true) || ...
    (strcmp (saved_db, evalc (['disp (db_params)'])) ~= true)
    [F, labels, entries, class] = LC_features (db_params.location, features_params);
    saved_features = evalc (['disp (features_params)']);
    saved_db = evalc (['disp (db_params)']);
    save('LC_features_and_labels.mat', 'F', 'labels', 'entries', 'saved_features', 'saved_db', '-v7.3');
end

%% learning and transformations
[F, kernels] = LC_learning (F, learning_params);

%% summarization
[F, labels, entries] = LC_summarization (F, labels, entries, summarization_params);
save('LC_data_summarized.mat', 'F', 'labels', 'entries', '-v7.3');

%% distribution equalization (in number of samples per class)
Nclass = length (unique (labels));

if strcmp (equalize_distribution, 'yes') == true
    [F, labels, entries] = LC_distribution_eq (F, labels, entries, Nclass);
end

%% classification

accv = zeros (Nfolds,1);
precv = zeros (Nfolds,1);
mapv = zeros (Nfolds,1);

total_samples = size (F, 2);
test_samples = floor (total_samples * tt_ratio);
train_samples = total_samples - test_samples;

for ifold = 1:Nfolds    
    fprintf ('classification fold %d\n', ifold);

    if strcmp (structured_validation, 'yes')
        [train_F, test_F, train_labels, test_labels, train_entries, test_entries] = LC_split_dataset (F, labels, entries, tt_ratio);
    else
        perm_idx = randperm(size (F, 2));
        test_idx = perm_idx(1 : test_samples)';
        train_idx = perm_idx(test_samples + 1:end)';

        test_labels = labels(test_idx);
        train_labels = labels(train_idx);
        test_F = F(:, test_idx);
        train_F = F(:, train_idx);    

        test_entries = entries(test_idx);
    end
    
    %%% OLS feature selection / dimensionality reduction
    if (dimensions ~= 0) 
        fprintf ('\tdim. reduction: ');
        [train_F, test_F, ~] = pls_multiclass_v2 (train_F, test_F, train_labels, Nclass, dimensions);
    end

   if strcmp (standardize, 'yes') == true %% independent
        fprintf ('\tstandardizing features...\n');
        [moys, stddevs, train_F] = LC_standardization (train_F);        

       numFrames = size (test_F, 2);
       test_F = (test_F - repmat (moys,1,numFrames))./repmat(stddevs,1,numFrames);
   end

    [accv(ifold), mapv(ifold)] = LC_classification (train_F, train_labels, test_F, test_labels, ...
        test_entries, Nclass,  classification_params);
end

acc = mean (accv); map = mean (mapv);

telapsed = toc (tstart);
fprintf ('\n');
results = sprintf ('[final results]\nacc = %f, map = %f (performance time: %f sec.)\n', acc, map, telapsed);
disp (results);

%% store results
LC_storage (all_p, results);

%% eof

