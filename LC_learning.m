function [Fm, kernels] = LC_learning (F, params)
    switch params.type
        case 'none'
            kernels = 0;
            Fm = F;
        case 'kmeans'
            disp ('unsupervised learning by kmeans...');
                
            if (strcmp (params.pca_whitening, 'yes') == true)
                fprintf ('\t\tapplying pca-whitening...\n');
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
            F = bases * F;

            end
            
            [~, kernels] = kmeans (F.', params.K); %, 'Distance', 'cosine');
            %kernels = SPKmeans (F.', params.K, 2);
            Fm = (kernels * F); %%% mapping
        otherwise
            error ('LifeClef2015 error: invalid learning');
    end 
end