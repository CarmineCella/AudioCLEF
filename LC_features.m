function [F, labels, entries, class] = LC_features (db_location, params)
F = [];
labels = [];
entries = [];
if (strcmp (params.scat_type, 'scat1'))
    opts{1}.time.size = params.scat_chunksize;
    opts{1}.time.T = params.scat_tw1;
    opts{1}.time.max_Q = params.scat_Q;
    opts{1}.time.gamma_bounds = [1 params.scat1_max_coeff];
    archs = sc_setup (opts);
end

if (strcmp (params.scat_type, 'scat2'))
    opts{1}.time.size = params.scat_chunksize;
    opts{1}.time.T = params.scat_tw1;
    opts{1}.time.max_Q = params.scat_Q;
    opts{2}.time.T = params.scat_tw2;
    opts{2}.time.handle = @morlet_1d;
    archs = sc_setup (opts);
end

if (strcmp (params.scat_type, 'scatj'))
    opts{1}.time.size = params.scat_chunksize;
    opts{1}.time.T = params.scat_tw1;
    opts{1}.time.max_Q = params.scat_Q;
    opts{2}.time.T = params.scat_tw2;
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
    error ('LifeClef2015 error: cannot find database');
end
foldernames = foldernames (3 : length(foldernames));

file_ctx = 1; % absolute file counter
class_ctx = 0;

class=zeros(0,1);
k=0;

for iFolder = 1 : length (foldernames)
    filenames = dir(strcat (db_location, '/', foldernames(iFolder).name));
    filenames = filenames (3 : length(filenames));
    % Skip hidden folder, e.g. .DS_Store on an OS X
    if foldernames(iFolder).name(1) == '.'
        continue
    end
    fprintf('processed folder %s (%d/%d)\n', ...
        foldernames (iFolder).name, iFolder, length (foldernames));
    class_ctx = class_ctx + 1;
    
    
    for iFile = 1 : length(filenames)

        filename = strcat (db_location, '/', foldernames(iFolder).name,'/',filenames(iFile).name);
        [~, ~, ext] = fileparts (filename);
        if (strcmp (ext, '.wav') == false)
            continue;
        end
        k= k + 1; class(k) = iFolder;
                
        [temp, sr] = audioread (filename);
        
        if (strcmp (params.rms_norm, 'yes') == true)
            fprintf ('\t\tapplying rms normalization...\n');
            temp = temp ./ sqrt (sum (temp .^2));
        end
        
        % feature extraction
        switch params.type
            case 'mfcc'
                fprintf ('\tcomputing mfcc on %s...\n', filename);
                [~, ff] = melfcc (temp, sr, 'maxfreq', params.mfcc_maxf, ...
                    'minfreq', params.mfcc_minf,'numcep', params.mfcc_ceps, 'nbands', params.mfcc_bands, ...
                    'fbtype', 'mel', 'dcttype', 2, 'wintime', params.mfcc_win, 'hoptime', params.mfcc_hop);
            case 'scattering'
                fprintf ('\tcomputing scattering on %s...\n', filename);
                S = sc_propagate(temp, archs);
                ff = sc_format(S);
            otherwise
                error ('LifeClef2015 error: invalid feature type');
        end
        
        origff = ff;
        if (strcmp (params.log_features, 'yes') == true)
            fprintf ('\t\tapplying log to features...\n');
            ff = log1p (epsilon_logS * max (ff, 0));
        end
        logff = ff;
        if (strcmp (params.thresholding, 'yes') == true)
            fprintf ('\t\tapplying thresholding...\n');
            ff = medfilt1 (ff, 7, size (ff, 1), 2);
        end
        threshff = ff;
        
        if (strcmp (params.debug, 'yes') == true)
            figure
            subplot (3, 1, 1)
            imagesc (origff)
            title ('original');
            subplot (3, 1, 2)
            imagesc (logff)
            title ('log');
            subplot (3, 1, 3);
            imagesc (threshff)
            title ('threshold');
            
            keyboard
            close all
        end
        
        for i = 1 : size (ff, 2)
            F = [F ff(:,i)];
            labels = [labels class_ctx];
            entries = [entries file_ctx];
        end
        
        file_ctx = file_ctx + 1; % absolute file counter
    end
end

fprintf ('\ntotal number of files: %d\n\n', file_ctx - 1);
end
