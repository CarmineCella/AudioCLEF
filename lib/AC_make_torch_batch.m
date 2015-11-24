function AC_make_torch_batch (F, labels, entries, params, crop_length)

if strcmp (params.type, 'none') == true
    return
end

disp ('exporting data to Torch...');

file = 'AC_torch_batch.h5';
testdir = 'torch_test';
traindir = 'torch_train';

nclasses =  numel(unique(labels));
        
[train_F, test_F, train_labels, test_labels, train_entries, test_entries] = AC_split_dataset (F, labels, entries, params.tt_ratio);
[moys, stddevs, train_F] = AC_standardization (train_F);

numFrames = size (test_F, 2);
test_F = (test_F - repmat (moys,1,numFrames))./repmat(stddevs,1,numFrames);

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
            label = lm_train (:, train_entries==utrain(i));
            filename = sprintf ('%d_features.h5', i);

            %features=features.';
            h5create(filename,'/features',size(features),'Datatype','double');
            h5write(filename,'/features', features);
            %save (filename, 'features');
            
            filename = sprintf ('%d_label.h5', i);
            h5create(filename,'/label', size(label),'Datatype','double');
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
            features = test_F(:, test_entries==utest(i));
            label = lm_test(:, test_entries==utest(i));
            filename = sprintf ('%d_features.h5', i);
            
            %features=features.';
            h5create(filename,'/features',size(features),'Datatype','double');
            h5write(filename,'/features', features);
            %save (filename, 'features');
            
            filename = sprintf ('%d_label.h5', i);
            h5create(filename,'/label', size(label),'Datatype','double');
            h5write(filename,'/label', label);
            %save (filename, 'label');
        end
        cd ..
        
        if exist ('AC_torch_metadata.h5', 'file')
            delete ('AC_torch_metadata.h5')
        end
        train_sz = [length(utrain) size(train_F, 1)];
        test_sz = [length(utest) size(test_F, 1)];
        h5create ('AC_torch_metadata.h5', '/train_sz', size(train_sz), 'Datatype', 'double');
        h5write ('AC_torch_metadata.h5', '/train_sz', train_sz);
        h5create ('AC_torch_metadata.h5', '/test_sz',size (test_sz), 'Datatype', 'double');
        h5write ('AC_torch_metadata.h5', '/test_sz', test_sz);        
        h5create('AC_torch_metadata.h5','/nclasses',size(nclasses), 'Datatype','double');
        h5write('AC_torch_metadata.h5','/nclasses', nclasses);        
    case 'frame_splitted'
        % train data
        if exist (traindir, 'dir')
            rmdir (traindir, 's')
        end
        mkdir (traindir)
        cd (traindir)        
        utrain = unique (train_entries);
        train_ctx = 1;
        for i = 1 : length (utrain)
            features = train_F (:, train_entries==utrain(i));
            label = lm_train (:, train_entries==utrain(i));
            disp(train_ctx)
            for j = 1 : crop_length :  size (features,2)
                filename = sprintf ('%d_features.h5', train_ctx);
                slice = features(:, j : j+crop_length-1);
                l = label(:,j:j+crop_length-1);
                h5create(filename,'/features',size(slice),'Datatype','double');
                h5write(filename,'/features', slice);
                %save (filename, 'features');
                
                filename = sprintf ('%d_label.h5', train_ctx);
                h5create(filename,'/label', size(l),'Datatype','double');
                h5write(filename,'/label', l);
                %save (filename, 'label');
                train_ctx = train_ctx + 1;
            end
        end        
        cd ..
        % test data
        if exist (testdir, 'dir')
            rmdir (testdir, 's')
        end
        mkdir (testdir)
        cd (testdir)
        utest = unique (test_entries);
        test_ctx = 1;
        for i = 1 : length (utest)
            features = test_F(:, test_entries==utest(i));
            label = lm_test(:, test_entries==utest(i));
            disp(test_ctx)
            for j = 1 : crop_length : size (features,2)
                filename = sprintf ('%d_features.h5', test_ctx);
                slice = features(:, j:j+crop_length-1);
                l = label(:,j:j+crop_length-1);
                h5create(filename,'/features',size(slice),'Datatype','double');
                h5write(filename,'/features', slice);
                %save (filename, 'features');
                
                filename = sprintf ('%d_label.h5', test_ctx);
                h5create(filename,'/label', size(l),'Datatype','double');
                h5write(filename,'/label', l);
                %save (filename, 'label');
                test_ctx = test_ctx + 1;
            end
        end
        cd ..
        
        if exist ('AC_torch_metadata.h5', 'file')
            delete ('AC_torch_metadata.h5')
        end
        train_sz = [train_ctx-1 size(train_F, 1)];
        test_sz = [test_ctx-1 size(test_F, 1)];
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
