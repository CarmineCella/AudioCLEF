function [Fm, kernels] = AC_learning (F, params)
Fc = F;
if (strcmp (params.pca_whitening, 'yes') == true)
    fprintf ('applying pca-whitening...\n');
    %     % from http://eric-yuan.me/ufldl-exercise-pca-image/
    %     x = F;
    %     xmean = mean(x, 1);
    %     x = bsxfun(@minus, x, xmean);
    %     xRot = zeros(size(x));
    %     [~, nfeatures] = size(x);
    %     sigma = x * x' ./ nfeatures;
    %     [U S V] = svd(sigma);
    %     xRot = U' * x;
    %     covar = zeros(size(x, 1));
    %     covar = xRot * xRot' ./ nfeatures;
    %     k = 0; % Set k to retain 99% of energy
    %     lambda = sum(S, 2);
    %     Sum = sum(lambda);
    %     temp = Sum;
    %     for id=size(lambda, 1):-1:1
    %         lambda(id);
    %         temp = temp - lambda(id);
    %         if (temp / Sum < 0.99)
    %             k = id;
    %             break;
    %         end
    %     end
    %     xHat = zeros(size(x));
    %     xHat = U(:, 1:k) * U(:, 1:k)' * x;
    %     epsilon = 0.1;
    %     xPCAWhite = zeros(size(x));
    %     xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U' * x;
    %
    %     xZCAWhite = zeros(size(x));
    %     xZCAWhite = U * xPCAWhite;
    %     F = xZCAWhite;
    
    bases = pca (F');
    Fc = bases * F;
end

if params.feature_percentile ~= 0
    fprintf ('applying feature percentile sparsification...\n');
    percentiles = prctile(F, params.feature_percentile, 2);
    below_threshold = bsxfun(@lt, F, percentiles);
    Fc = F;
    Fc(below_threshold) = 0;
end
switch params.type
    case 'none'
        kernels = 0;
        Fm = F;
    case 'kmeans'
        disp ('unsupervised learning by kmeans...');
        [~, ~, Fc] = AC_standardization(Fc); % standardization is mandatory!
        [~, kernels] = kmeans (Fc.', params.K); %, 'Distance', 'cosine');
        %Fm = (kernels * F); %%% mapping
        nKernels = size(kernels, 1);
        permutation = randperm(nKernels)
        kernels = kernels(permutation, :);
        Fm = conv2(F, kernels);
    case 'nnmf'
        disp ('unsupervised learning by nnmf...');
        [w, h] = nnmf (Fc, params.K);
        %Fm = h; %%% mapping is directly given by activation functions
        kernels = w;
        Fm = conv2 (F, w);
    otherwise
        error ('AudioCLEF error: invalid learning');
end
end
