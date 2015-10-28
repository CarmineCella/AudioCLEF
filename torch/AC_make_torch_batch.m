function AC_make_torch_batch (F, labels, entries, params)

file = 'AC_torch_batch.h5';
testdir = 'torch_test';
traindir = 'torch_train';

nclasses =  numel(unique(labels));
        
[~, ~, F] = AC_standardization(F); % standardization is mandatory!

if (params.dimensions ~= 0)
    F_orig = F;
    [mu,E,V] = AC_pca (F');
    M = dimensions; 
    F = (V(:,1:M)')*(F-repmat(mu',[1 size(F,2)])); 
end


[train_F, test_F, train_labels, test_labels, train_entries, test_entries] = AC_split_dataset (F, labels, entries, params.tt_ratio);

switch params.type
    case 'label'
        lm_train = train_labels;
        lm_test = test_labels;
    case 'probability'
        Nclasses=numel(unique(train_labels));
        lm_train = zeros(Nclasses, numel(train_labels));
        
        for i = 1:numel(train_labels)
            lm_train(train_labels(i),i)=1;
        end
        
        Nclasses=numel(unique(test_labels));
        lm_test = zeros(Nclasses, numel(test_labels));
        
        for i = 1:numel(test_labels)
            lm_test(test_labels(i),i)=1;
        end
end

switch params.mode
    case 'monolithic'
        if exist (file, 'file')
            delete (file)
        end
        h5create(file,'/train_F',size(train_F),'Datatype','double');
        h5write(file,'/train_F', train_F);
        h5create(file,'/train_labels',size(lm_train),'Datatype','double');
        h5write(file,'/train_labels', lm_train);
        h5create(file,'/train_entries',size(train_entries),'Datatype','double');
        h5write(file,'/train_entries', train_entries);

        h5create(file,'/test_F',size(test_F),'Datatype','double');
        h5write(file,'/test_F', test_F);
        h5create(file,'/test_labels',size(lm_test),'Datatype','double');
        h5write(file,'/test_labels', lm_test);
        h5create(file,'/test_entries',size(test_entries),'Datatype','double');
        h5write(file,'/test_entries', test_entries);

        h5create(file,'/nclasses',size(nclasses), 'Datatype','double');
        h5write(file,'/nclasses', nclasses);
    case 'splitted'
        % train data
        if exist (traindir, 'dir')
            rmdir (traindir, 's')
        end
        mkdir (traindir)
        cd (traindir)        
        utrain = unique (train_entries);
        for i = 1 : length (utrain)
            features = train_F (:, train_entries==utrain(i));
            label = mean (lm_train (:, train_entries==utrain(i)), 2);

            filename = sprintf ('%d_features.h5', i);
            h5create(filename,'/features',size(features),'Datatype','double');
            h5write(filename,'/features', features);            
            %save (filename, 'features');
             
            filename = sprintf ('%d_label.h5', i);
            h5create(filename,'/label',size(label),'Datatype','double');
            h5write(filename,'/label', label);            
            %save (filename, 'label');
        end        
        cd ..
        % test data
        if exist (testdir, 'dir')
            rmdir (testdir, 's')
        end
        mkdir (testdir)
        cd (testdir)
        utest = unique (test_entries);
        for i = 1 : length (utest)
            features = test_F (:, test_entries==utest(i));
            label = mean (lm_train (:, test_entries==utest(i)), 2);

            filename = sprintf ('%d_features.h5', i);
            h5create(filename,'/features',size(features),'Datatype','double');
            h5write(filename,'/features', features);            
            %save (filename, 'features');
            
            filename = sprintf ('%d_label.h5', i);
            h5create(filename,'/label',size(label),'Datatype','double');
            h5write(filename,'/label', label);     
            %save (filename, 'label');
        end
        cd ..
        
        if exist ('AC_torch_metadata.h5', 'file')
            delete ('AC_torch_metadata.h5')
        end
        train_sz = size (train_F);
        test_sz = size (test_F);
        h5create ('AC_torch_metadata.h5', '/train_sz', size(train_sz), 'Datatype', 'double');
        h5write ('AC_torch_metadata.h5', '/train_sz', train_sz);
        h5create ('AC_torch_metadata.h5', '/test_sz',size (test_sz), 'Datatype', 'double');
        h5write ('AC_torch_metadata.h5', '/test_sz', test_sz);        
        h5create('AC_torch_metadata.h5','/nclasses',size(nclasses), 'Datatype','double');
        h5write('AC_torch_metadata.h5','/nclasses', nclasses);
    otherwise
        error ('AudioCLEF: invalid export mode for Torch');
end

end
