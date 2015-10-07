function [train_F, test_F, train_labels, test_labels, train_entries, test_entries] = AC_split_dataset (F, labels, entries, tt_ratio)

fprintf ('\tsplitting train/test...');
train_F = [];
test_F = [];
train_labels = [];
test_labels = [];
train_entries = [];
test_entries = [];

uni_labels = unique(labels);
for icl=1:numel(uni_labels)
    class_F = F(:, labels==uni_labels(icl));
    class_entries = entries(labels==uni_labels(icl));
    class_ids = unique (class_entries);
    
    files_in_class = numel (unique(class_entries));
    rand_idx = randperm (files_in_class);
    train_samples = floor (files_in_class * (1. - tt_ratio));
    test_samples = files_in_class - train_samples;
    train_idx = rand_idx (1:train_samples)';
    test_idx = rand_idx (train_samples + 1 : end)';
    
    mapped_test_entries = class_ids(test_idx);
    for te_file=1:numel(mapped_test_entries)
        extracted_tmp = class_entries(class_entries == mapped_test_entries(te_file));
        test_entries = [test_entries extracted_tmp];
        test_F = [test_F class_F(:, class_entries == mapped_test_entries(te_file))];
        test_labels = [test_labels ones(1, numel(extracted_tmp))*uni_labels(icl)];
    end
    
    mapped_train_entries = class_ids(train_idx);
    for tr_file=1:numel(mapped_train_entries)
        extracted_tmp = class_entries(class_entries == mapped_train_entries(tr_file));
        train_entries = [train_entries extracted_tmp];
        train_F = [train_F class_F(:, class_entries == mapped_train_entries(tr_file))];
        train_labels = [train_labels ones(1, numel(extracted_tmp))*uni_labels(icl)];
    end
    
end
    fprintf ('%d vs %d samples\n', size (train_F, 2), size (test_F, 2));
end