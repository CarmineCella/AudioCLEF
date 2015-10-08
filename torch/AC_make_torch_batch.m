function AC_make_torch_batch (file, F, labels, entries, tt_ratio)
 
[train_F, test_F, train_labels, test_labels, train_entries, test_entries] = AC_split_dataset (F, labels, entries, tt_ratio);

h5create(file,'/train_F',size(train_F),'Datatype','double');
h5write(file,'/train_F', train_F);
h5create(file,'/train_labels',size(train_labels),'Datatype','double');
h5write(file,'/train_labels', train_labels);
h5create(file,'/train_entries',size(train_entries),'Datatype','double');
h5write(file,'/train_entries', train_entries);

h5create(file,'/test_F',size(test_F),'Datatype','double');
h5write(file,'/test_F', test_F);
h5create(file,'/test_labels',size(test_labels),'Datatype','double');
h5write(file,'/test_labels', test_labels);
h5create(file,'/test_entries',size(test_entries),'Datatype','double');
h5write(file,'/test_entries', test_entries);

end
