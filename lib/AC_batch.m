
function [F, labels, entries, acc, map, kernels] = AC_batch (db_params, features_params, learning_params, summarization_params, ...
    equalization_params, batch_params, classification_params)
saved_features = ''; % to decide if features must be recomputed
saved_db = '';
all_p = sprintf ('db_params:\n%s\nfeatures_params:\n%s\nlearning_params:\n%s\nsummarization_params:\n%s\nequalization_params:\n%s\nclassification_params:\n%s\nbatch_params:\n%s\n', ...
    evalc (['disp (db_params)']), evalc (['disp (features_params)']), evalc (['disp (learning_params)']), ...
    evalc (['disp (summarization_params)']), evalc (['disp (equalization_params)']), evalc (['disp (classification_params)']), evalc (['disp (batch_params)']));

%% compute features
tstart = tic;
if exist('AC_features_and_labels.mat', 'file') == 2
    load('AC_features_and_labels.mat');
end

if (strcmp (saved_features, evalc (['disp (features_params)'])) ~= true) || ...
        (strcmp (saved_db, evalc (['disp (db_params)'])) ~= true)
    [F, labels, entries] = AC_features (db_params.location, features_params);
    saved_features = evalc (['disp (features_params)']);
    saved_db = evalc (['disp (db_params)']);
    save('AC_features_and_labels.mat', 'F', 'labels', 'entries', 'saved_features', 'saved_db', '-v7.3');
end

%% distribution equalization (in number of samples per class)
Nclass = length (unique (labels));
[F, labels, entries] = AC_distribution_eq (F, labels, entries, Nclass, equalization_params);

%% learning    
kernels = AC_learning (F, learning_params);
    

%% summarization
[F, labels, entries] = AC_summarization (F, labels, ...
    entries, summarization_params);

%% classification
accv = zeros (batch_params.Nfolds,1);
mapv = zeros (batch_params.Nfolds,1);

for ifold = 1:batch_params.Nfolds
    fprintf ('classification fold %d\n', ifold);
    
    if strcmp (batch_params.structured_validation, 'yes')
        [train_F, test_F, train_labels, test_labels, train_entries, test_entries] = ...
            AC_split_dataset (F, labels, entries, batch_params.tt_ratio);
    else
        total_samples = size (F, 2);
        test_samples = floor (total_samples * params.tt_ratio);
        %train_samples = total_samples - test_samples;
        
        perm_idx = randperm(size (F, 2));
        test_idx = perm_idx(1 : test_samples)';
        train_idx = perm_idx(test_samples + 1:end)';
        
        test_labels = labels(test_idx);
        train_labels = labels(train_idx);
        test_F = F(:, test_idx);
        train_F = F(:, train_idx);
        
        test_entries = entries(test_idx);
    end
    
    %%% transformations
    if kernels~= 0
        train_F = (kernels * train_F); %%% mapping
        test_F = (kernels * test_F); %%% mapping
    end
    
   
    %%% OLS feature selection / dimensionality reduction
    if (batch_params.dimensions ~= 0)
        fprintf ('\tdim. reduction: ');
        [train_F, test_F, ~] = pls_multiclass_v2 (train_F, test_F, train_labels, ...
            Nclass, params.dimensions);
    end
    
    %%% standardization
    if strcmp (batch_params.standardize, 'yes') == true %% independent
        fprintf ('\tstandardizing features...\n');
        [moys, stddevs, train_F] = AC_standardization (train_F);
        
        numFrames = size (test_F, 2);
        test_F = (test_F - repmat (moys,1,numFrames))./repmat(stddevs,1,numFrames);
    end
    
    [accv(ifold), mapv(ifold)] = AC_classification (train_F, train_labels, ...
        test_F, test_labels, test_entries, Nclass, classification_params);
end

acc = mean (accv); map = mean (mapv);

telapsed = toc (tstart);
fprintf ('\n');
results = sprintf ('[final results]\nacc = %f, map = %f, (performance time: %f sec.)\n', ...
    acc, map, telapsed);
disp (results);

%% store results
AC_storage (all_p, results);

end

%% eof

