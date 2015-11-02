function [Fs, labels_s, entries_s] = AC_summarization (F, labels, entries, params)

if strcmp(params.type, 'none')
    Fs = F;
    labels_s = labels;
    entries_s = entries;
    return
end

usedFiles = unique (entries);
nFiles = numel (usedFiles);
labels_s = zeros(1, nFiles);
entries_s = zeros(1, nFiles);

file_features = cell(nFiles, 1);
for file_index = 1:nFiles
    selection = (entries == usedFiles (file_index));
    file_features{file_index} = F(:, selection);
    gi = labels(selection);

    labels_s(file_index) = unique(gi);
    entries_s(file_index) = file_index;
end
nFeatures = size(F, 1);

switch (params.type)
    case 'mean_std'
        fprintf ('\tsummarizing by mean and std deviation...\n');
        Fs = zeros(2, nFeatures, nFiles);
        for file_index = 1:nFiles
            Fs(1, :, file_index) = mean(file_features{file_index}, 2);
            Fs(2, :, file_index) = std(file_features{file_index}, 0, 2);
        end
        Fs = reshape(Fs, 2 * nFeatures, nFiles);
     case 'mean'
        disp ('summarizing by mean...');
        Fs = zeros(1, nFeatures, nFiles);
        for file_index = 1:nFiles
            Fs(1, :, file_index) = mean(file_features{file_index}, 2);
        end
        Fs = reshape(Fs, nFeatures, nFiles);
     case 'k-means'
        disp ('summarizing by k-means...');
        Fs = zeros(params.components, nFeatures, nFiles);       
        for file_index = 1:nFiles          
            [~, C] = kmeans (file_features{file_index}', params.components);
            for cluster_index = 1:params.components
                Fs(cluster_index, :, file_index) = C(cluster_index, :)';
            end
        end
        Fs = reshape(Fs, params.components * nFeatures, nFiles);        
    case 'scat_summary'
        disp('summarizing with scattering...');
        opts{1}.time.T = 512;
        archs = sc_setup(opts);
        % Total number of feature made by phi + psi_s
        nPaths = 1 + length(archs{1}.banks{1}.psis{1});
        Fs = zeros(nFeatures, nPaths, nFiles);
        for file_index = 1:nFiles
            U0 = initialize_U(file_features{file_index}.', archs{1}.banks{1});
            Y1 = U_to_Y(U0, archs{1});
            U1 = Y_to_U(Y1{end}, archs{1});
            if ismatrix(U0.data)
                % There is only one chunk. Layout is time x feature
                % We sum along dimension 1 (time)
                for path_index = 1:(nPaths-1)
                    Fs(:, path_index, file_index) = ...
                        squeeze(sum(U1.data{path_index}, 1));
                end
                Fs(:, end, file_index) = squeeze(sum(U0.data, 1));
            elseif ndims(U0.data)==3
                % There are chunks. Layout is time x chunk x feature
                for path_index = 1:(nPaths-1)
                    Fs(:, path_index, file_index) = ...
                        squeeze(sum(sum(U1.data{path_index}, 1), 2));
                end
                Fs(:, end, file_index) = squeeze(sum(sum(U0.data, 1), 2));
            end
        end
        Fs = reshape(Fs, nFeatures * nPaths, nFiles);
    otherwise
        error ('AudioCLEF error: invalid summarization');
end
end
