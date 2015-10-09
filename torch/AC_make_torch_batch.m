function AC_make_torch_batch (file, F, labels, entries, tt_ratio)
 
if exist (file, 'file')
    delete (file)
end

[train_F, test_F, train_labels, test_labels, train_entries, test_entries] = AC_split_dataset (F, labels, entries, tt_ratio);

lm1 = train_labels;
lm2 = test_labels;

% Nclasses=numel(unique(train_labels));
% lm1 = zeros(Nclasses, numel(train_labels));
% 
% for i = 1:numel(train_labels)
%     lm1(train_labels(i),i)=1;
% end
% 
% Nclasses=numel(unique(test_labels));
% lm2 = zeros(Nclasses, numel(test_labels));
% 
% for i = 1:numel(test_labels)
%     lm2(test_labels(i),i)=1;
% end

subplot(211)
imagesc (lm1)
subplot(212)
imagesc (lm2)
figure

h5create(file,'/train_F',size(train_F),'Datatype','double');
h5write(file,'/train_F', train_F);
h5create(file,'/train_labels',size(lm1),'Datatype','double');
h5write(file,'/train_labels', lm1);
h5create(file,'/train_entries',size(train_entries),'Datatype','double');
h5write(file,'/train_entries', train_entries);

h5create(file,'/test_F',size(test_F),'Datatype','double');
h5write(file,'/test_F', test_F);
h5create(file,'/test_labels',size(lm2),'Datatype','double');
h5write(file,'/test_labels', lm2);
h5create(file,'/test_entries',size(test_entries),'Datatype','double');
h5write(file,'/test_entries', test_entries);

plot (test_labels)

end
