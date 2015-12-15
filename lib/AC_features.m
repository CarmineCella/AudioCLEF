function [F, labels, entries] = AC_features(db_location, params)
if (strcmp (params.scat_type, 'scat1'))
    opts{1}.time.T = params.scat_tw;
    opts{1}.time.max_Q = params.scat_Q;
    opts{1}.time.gamma_bounds = [1 params.scat1_max_coeff];
    archs = sc_setup (opts);
end

if (strcmp (params.scat_type, 'scat2'))
    opts{1}.time.T = params.scat_tw;
    opts{1}.time.max_Q = params.scat_Q;
    opts{2}.time.T = params.scat_tw;
    opts{2}.time.handle = @morlet_1d;
    archs = sc_setup (opts);
end

if (strcmp (params.scat_type, 'scatj'))
    opts{1}.time.T = params.scat_tw;
    opts{1}.time.max_Q = params.scat_Q;
    opts{2}.time.T = params.scat_tw;
    opts{2}.time.handle = @morlet_1d;
    opts{2}.gamma.phi_bw_multiplier = 1;
    opts{2}.gamma.T = 2 * opts{1}.time.max_Q;
    % when chunking, the chunk variable is at subscript 2 so the gamma variable
    % (the one we transform in joint scattering) is at subscript 3
    opts{2}.gamma.subscripts = 3;
    archs = sc_setup (opts);
end

epsilon_logS = 1e-2; %2^(-20); % same as scatnet

foldernames = dir(db_location);
if (length (foldernames) < 3)
    error ('AudioCLEF error: cannot find database');
end
foldernames = foldernames (3:length(foldernames));
nFolders = length(foldernames);

folder_features = cell(nFolders, 1);
folder_entries = cell(nFolders, 1);
folder_labels = cell(nFolders, 1);

file_ctx = 0; % absolute file counter (incremented by nNonempty_files, not by 1)
folder_ctx = 0; % absolute folder counter

sobv = [-1 0 1 ;
        -2 0 2 ;
        -1 0 1];
    
sizes  = [];
for iFolder = 1 : length (foldernames)
    filenames = dir(strcat (db_location, '/', foldernames(iFolder).name));
    filenames = filenames (3 : length(filenames));
    % Skip hidden folder, e.g. .DS_Store on an OS X
    if foldernames(iFolder).name(1) == '.'
        continue
    end
    % Skip empty folder
    if length(filenames) < 1
        continue
    end
    folder_ctx = folder_ctx + 1;
    fprintf('processed folder %s (%d/%d)\n', ...
        foldernames (iFolder).name, iFolder, length (foldernames));
    
    nFiles = length(filenames);
    file_features = cell(nFiles, 1);
    
    for iFile = 1 : length(filenames)
        filename = strcat (db_location, '/', foldernames(iFolder).name, ...
            '/',filenames(iFile).name);
        [~, ~, ext] = fileparts (filename);
        if (strcmp (ext, '.wav') == false)
            continue;
        end
        
        [temp, sr] = audioread (filename);
        
        if (strcmp (params.rms_norm, 'yes') == true)
            fprintf ('\tapplying rms normalization...\n');
            temp = temp ./ sqrt (sum (temp .^2)) .* sqrt (length (temp));
        end
        
        % feature extraction
        switch params.type
            case 'mfcc'
                fprintf ('\tcomputing mfcc on %s...\n', filename);
                [~, file_features{iFile}] = ...
                    melfcc (temp, sr, 'maxfreq', params.mfcc_maxf, ...
                    'minfreq', params.mfcc_minf,'numcep', params.mfcc_ceps, ...
                    'nbands', params.mfcc_bands, 'fbtype', 'mel', ...
                    'dcttype', 2, 'wintime', params.mfcc_win, ...
                    'hoptime', params.mfcc_hop);
                %file_features{iFile} = conv2 (file_features{iFile}, sobv);
            case 'scattering'
                if strcmp (params.scat_norm, 'yes')
                    fprintf ('\tcomputing normalized scattering on %s...\n', filename);
                    S = sc_propagate_renorm(temp, archs);
                else
                    fprintf ('\tcomputing scattering on %s...\n', filename);
                    S = sc_propagate(temp, archs);
                end
                [file_features{iFile}] = sc_format(S);
            case 'alogc'
                fprintf('\tcomputing average-log coefficients on %s...\n', filename);
                [~, file_features{iFile}] = ...
                    AC_AverageLogCoeff(temp, params.alogc_win, ...
                    params.alogc_olap, params.alogc_nbands, ...
                    params.alogc_ncoeff, params.alogc_alpha);
            otherwise
                error ('AudioCLEF error: invalid feature type');
        end
        
        if (strcmp (params.log_features, 'yes') == true)
            fprintf ('\t\tapplying log to features...\n');
            file_features{iFile} = ...
                log1p (epsilon_logS * max (file_features{iFile}, 0));
        end
        if (strcmp (params.thresholding, 'yes') == true)
            fprintf ('\t\tapplying thresholding...\n');
            file_features{iFile} = medfilt1(file_features{iFile}, 7, ...
                size(file_features{iFile}, 1), 2);
        end
        if (params.nCrops ~= 0)
            F_in = file_features{iFile};
            [nFeatures, nFrames] = size(F_in);
            switch params.detection_function
                case 'spectrum_energy'
                    assert(strcmp(params.type, 'mfcc'));
                    detection_function = sum(F_in, 1);
                case 'spectrum_flux'
                    assert(strcmp(params.type, 'mfcc'));
                    detection_function = ...
                        sum(abs(diff(F_in, 1)));
                case 'scattering_flux'
                    assert(strcmp(params.type, 'scattering'));
                    second_layer = formatted_layers{1+2};
                    detection_function = sum(second_layer, 1);
                case 'scattering_energy'
                    assert(strcmp(params.type, 'scattering'))
                    detection_function = sum(F_in,12);
            end
            if nFrames <= params.crop_length
                nReplications = ceil(params.crop_length / nFrames);
                F_in = repmat(F_in, 1, nReplications);
                detection_function = ...
                    repmat(detection_function, 1, nReplications);
                nFrames = size(F_in, 2);
            end
            half_crop_length = round(params.crop_length / 2);  
            [peaks, locations] = findpeaks(detection_function, ...
                'SortStr', 'descend', ...
                'MinPeakDistance', params.crop_length, ...
                'NPeaks', params.nCrops) ;
            F_out = zeros(nFeatures, params.crop_length, params.nCrops);
            for crop_index = 1:params.nCrops
                peak_index = mod(crop_index - 1, length(locations)) + 1;
                location = locations(peak_index);
                start = location - half_crop_length;
                stop = location + half_crop_length - 1;
                range = start:stop;
                mod_range = mod(range - 1, nFrames) + 1;
                F_out(:, :, crop_index) = F_in(:, mod_range);
            end
            % backward-comptability to get 2d output
            F_out = ...
                reshape(F_out, [nFeatures, params.crop_length * params.nCrops]);
            subplot(311)
            findpeaks(detection_function, ...
                'SortStr', 'descend', ...
                'MinPeakDistance', params.crop_length, ...
                'NPeaks', params.nCrops) ;
            subplot(312);
            imagesc (log(1e3 * F_in));
            hold on;
            for peak_index = 1:length(locations)
                location = locations(peak_index);
                start = location - half_crop_length;
                stop = location + half_crop_length - 1;
                mod_start = mod(start - 1, nFrames) + 1;
                mod_stop = mod(stop - 1, nFrames) + 1;
                line([mod_start, mod_start], [1, nFeatures], 'Color', 'r');
                line([mod_stop, mod_stop], [1, nFeatures], 'Color', 'k');
            end
            hold off;
            subplot (313)
            imagesc (log(1e3 * F_out));
            set(gcf, 'WindowStyle', 'docked')
            file_features{iFile} = F_out;
        end
        
        sizes = [sizes size(file_features{iFile}, 2)];
    end

    % Get rid of empty files
    empty_files = cellfun(@isempty, file_features);
    nonempty_files = ~empty_files;
    folder_features{iFolder} = [file_features{~empty_files}];
    
    % Get number of frames per non-empty file
    nNonempty_files = sum(nonempty_files);
    nFrames_per_file = cellfun(@(x) size(x,2), file_features(nonempty_files));
    
    % Retrieve file entries
    single_file_entries = file_ctx + (1:nNonempty_files).';
    file_entries = arrayfun(@(x, y) repmat(x, y, 1), single_file_entries, ...
        nFrames_per_file, 'UniformOutput', false);
    file_entries = cellfun(@transpose, file_entries, 'UniformOutput', false);
    folder_entries{iFolder} = [file_entries{:}];
    
    % Retrieve file labels
    single_file_labels = ones(nNonempty_files, 1) * folder_ctx;
    file_labels = arrayfun(@(x, y) repmat(x, y, 1), single_file_labels, ...
        nFrames_per_file, 'UniformOutput', false);
    file_labels = cellfun(@transpose, file_labels, 'UniformOutput', false);
    folder_labels{iFolder} = [file_labels{:}];
    
    file_ctx = file_ctx + nNonempty_files;
end

figure
stem (sizes);
hold on
plot (median(sizes) .* ones(1, size(sizes,2)))
plot (mean(sizes) .* ones(1, size(sizes,2)))
title ('Distribution of frames in dataset')

% Get rid of empty folders
empty_folders = cellfun(@isempty, folder_features);
nonempty_folders = ~empty_folders;

F = [folder_features{nonempty_folders}];
labels = [folder_labels{nonempty_folders}];
entries = [folder_entries{nonempty_folders}];

if params.feature_percentile ~= 0
    fprintf ('\tapplying feature percentile sparsification...\n');
    percentiles = prctile(F, params.feature_percentile, 2);
    below_threshold = bsxfun(@lt, F, percentiles);
    Fc = F;
    Fc(below_threshold) = 0;
end

fprintf ('\ntotal number of files: %d\n\n', file_ctx);
end
