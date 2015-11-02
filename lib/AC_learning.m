function kernels = AC_learning (F, params)
Fc = F;
if (strcmp (params.pca_whitening, 'yes') == true)
    fprintf ('\tapplying pca-whitening...\n');
    bases = pca (F');
    Fc = bases * F;
end

switch params.type
    case 'none'
        kernels = 0;
    case 'kmeans'
        fprintf ('\tkmeans feature learning...\n');
        [~, ~, Fc] = AC_standardization(Fc); % standardization is mandatory!
        [~, kernels] = kmeans (Fc.', params.K); %, 'Distance', 'cosine');
    otherwise
        error ('AudioCLEF error: invalid learning');
end
end
