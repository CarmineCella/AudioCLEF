function [acc, map, cmat] = LC_classification (train_F, train_labels, test_F, test_labels, ...
    test_entries, Nclass, params)
switch params.type
    case 'SVM'
        fprintf ('\tsvm: ');
        svm_opt = sprintf('-s 0 -t %d -h 0 -b 1 -g %f -c %f', params.svm_kernel, params.svm_gamma, params.svm_C);
        % convert features to single precision (required by LIBSVM)
        %train_F = single(train_F);
        %test_F = single(test_F);
%         model = svmtrain1(train_labels', train_F', svm_opt);
%         [~,  ~, probs] = svmpredict1(test_labels', test_F', model, '-b 1');
        
        %kernel_train = kernelmatrix('rbf',train_F, [], params.svm_gamma);
        %kernel_test = kernelmatrix('rbf',train_F,test_F, params.svm_gamma);
        kernel_train = kernelmatrix('lin',train_F');
        kernel_test = kernelmatrix('lin',train_F',test_F');
        
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
            'Method', 'classification');
        [~, probs] = model.predict (test_F');
    otherwise
        error ('LifeClef2015 error: invalid classification method');
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
    fprintf ('\tframe-wise classification...\n');
    predicted_labels = predicted_matrix(:, 1);
    predicted_files = predicted_matrix;
    ground_truth = test_labels;  
end

perf = classperf (ground_truth, predicted_files(:, 1));
acc = perf.CorrectRate;
map = LC_MAP_at_K (ground_truth, predicted_files, Nclass);
cmat = confusionmat(ground_truth, predicted_files(:, 1));

fprintf ('\taccuracy: %f, map: %f\n', acc, map);

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
