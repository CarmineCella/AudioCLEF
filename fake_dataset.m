function [F, labels, entries] = fake_dataset (nfeatures, nsamples)
     F = ones(nfeatures, nsamples) .* (rand(nfeatures, nsamples) .* -1);
     F1 = ones(nfeatures, nsamples) .* rand (nfeatures, nsamples);
     F(:, [1:2:end]) = F1(:, [1:2:end]);
     
     labels=ones(1, nsamples) .* 2;
     labels1=ones(1, nsamples);
     labels(1:2:end) = labels1(1:2:end);
     
     entries=[1:1:nsamples];
end
