function [acc, map, cmat] = AC_classification (train_F, train_labels, ...
    test_F, test_labels, test_entries, Nclass, params)

if strcmp (params.type, 'none')
    acc  = 0;
    map = 0;
    cmat = 0;
    return
end

switch params.type
    case 'SVM'
        fprintf ('\tsvm: ');

        sigma = mean(sqrt(sum(train_F.^2,1)))*params.svm_sigma_v;
        
        if strcmp (params.svm_kernel, 'rbf') == true
            kernel_train = kernelmatrix(params.svm_kernel,train_F, [], sigma);
            kernel_test = kernelmatrix(params.svm_kernel,train_F,test_F, sigma);
        else
            kernel_train = kernelmatrix('lin',train_F);
            kernel_test = kernelmatrix('lin',train_F,test_F);
        end
        [kernel_train,kernel_test]=prepare_kernel_for_svm(kernel_train,kernel_test);
        [~, probs]=SVM_1vsALL_wrapper(train_labels, test_labels, kernel_train,kernel_test,params.svm_C);
        probs=probs';
    case 'RF'
        % parse labels into strings with leading zeros
        nTraining_samples = length(train_labels);
        train_labels_s = cell(1,nTraining_samples);
        for training_index = 1:nTraining_samples
            train_label = train_labels(training_index);
            train_labels_s{training_index} = sprintf('%0.4d', train_label);
        end
        fprintf ('\trandom forest...\n');
        model = TreeBagger(params.RF_ntree, train_F', train_labels_s, ...
            'Method', 'classification', 'oobvarimp','on', 'Fboot', params.RF_fboot);
        [~, probs] = model.predict (test_F');
        num_oob_per_tree = sum(sum(model.OOBIndices))/params.RF_ntree;
        fprintf('\tin-bag samples: %d/%d\n', floor(num_oob_per_tree), size(train_F',1))
        
    otherwise
        error ('AudioCLEF error: invalid classification method');
end

[~, predicted_matrix] = sort(probs, 2, 'descend');
boolean_matrix = bsxfun(@eq, predicted_matrix, test_labels.');

if (strcmp (params.histogram_voting, 'yes') == true)
    fprintf ('\thistogram voting...\n');
    %...........................................................................
    ground_truth = [];
    file_list = [];
    uni_test_entries=unique(test_entries,'stable');
    for te_file=1:numel(uni_test_entries)
        current_file = uni_test_entries(te_file);
        file_list = [file_list current_file];
        associated_label = mean (test_labels(test_entries==current_file));
        ground_truth = [ground_truth associated_label];
    end
    
    predicted_files = zeros (length (ground_truth), size (predicted_matrix, 2));
    for j = 1:size (predicted_matrix, 2)
        predicted_labels = predicted_matrix(:, j);
        predicted_file_label = [];
        for te_file = 1:numel(uni_test_entries)
            current_file = uni_test_entries(te_file);
            votes = hist(predicted_labels(test_entries==current_file),(1:Nclass)');
            [~, pf] = max(votes);
            predicted_file_label = [predicted_file_label pf];
        end
        predicted_files (:, j) = predicted_file_label;
    end
else
    % NB: this version does not take into account a global voting per file
    % in case of no summarization applied
    fprintf ('\telement-wise classification...\n');
    predicted_labels = predicted_matrix(:, 1);
    predicted_files = predicted_matrix;
    ground_truth = test_labels;
end

perf = classperf (ground_truth, predicted_files(:, 1));
acc = perf.CorrectRate;
map = AC_MAP_at_K (ground_truth, predicted_files, Nclass);
cmat = confusionmat(ground_truth, predicted_files(:, 1));

fprintf ('\taccuracy: %f, map: %f\n', acc, map);
if strcmp  (params.debug, 'yes') == true
    figure
    subplot (3, 1, 1)
    imagesc (boolean_matrix)
    title ('Boolean matrix');
    subplot (3, 1, 2)
    plot (ground_truth)
    hold on
    plot (predicted_files(:,1), 'r')
    title ('Test vs predicted')
    subplot (3, 1, 3)
    imagesc (cmat);
    title ('Confusion matrix');
end
end
