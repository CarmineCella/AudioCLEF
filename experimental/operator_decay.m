%% load and reshape data
%close all
load '../torch/l1_weights_cc.mat';

d1 = 40;
d2 = 40;
families = 2;
alpha = .5;
k = x';
k_r = reshape (k, d1, 15, d2);

%% calculate 2D fft by padding
k_p = pad_signal (k_r, [d1 50 d2], 'zero', 0);
K = fft2 (k_p);
aK = abs (K);
figure
for i = 1 : d2
    subplot (d2/10, 10, i)
    imagesc ( (fftshift (  (aK(:, :, i)))))
end
%%
dist = zeros (d2, d2);
for i = 1 : d2
   for j = 1 : d2
       d = (abs (aK (:, :, i)- aK (:, :, j)));
      dist (i,j) = sum (sum (d));
   end
end
figure
imagesc (dist ./ max (max(dist)))

maK = squeeze (sum (aK))';
dif = squareform (pdist (maK));
figure 
imagesc (dif)

[yhisto, xhisto] = hist (dist(:), d2);
var = (sum (yhisto .* xhisto / sum (yhisto))) * .9;
A = exp (-(dif.^2)/(var^2));
szadj = size (A);
D = zeros (szadj(1), szadj(1));

for i = 1 : szadj(1)
    D (i,i) = sum (A (i,:));
end

L = (D^(-alpha)) * A * (D^(-alpha));

figure
imagesc (L)

[u,d] = eig(L);
[evals, ma]= sort (diag(d), 'descend');
U = u(:, ma);
principal_eig = [U(:,2)  U(:,3)];
[idx1, C1] = kmeans (principal_eig, families);
 
figure
scatter (principal_eig(:,1), principal_eig(:, 2),[], idx1)

%% calculate spectrum of the operator
clear t;
for i = 1 : size (K, 2);
    t(i, :) = eig(squeeze (K(:, i, :))' * squeeze (K(:, i, :)));
end
t = abs (t (:)); %log (abs (t(:)));
figure
plot (sort (t, 'descend'))
%% build the operator T

% Assume our input signals are represented as matrices s, such that s_{j,r}
% represents the value of coordinate j at time r
% Assume that our output signals are represented as matrices Ts,  such that
% (Ts)_{i,q} is the value of coordinate i at time q
% T is represented by a tensor T_{i,j,q,r} such that:
% (Ts)_{i,q} = \sum_{j} \sum_{r} T_{i,j,q,r} s_{j,r}
% T can be constructed from the filters in k (of size K*(KW)*K') in the
% following manner:
% length_output = 46;
% length_input = 50;
% dW = 1;
% 
% T = zeros(d2,d1,length_output,length_input); % warning, can become very large
% 
% for i = 1:d2
%     for q = 1:length_output
%        % fprintf(['Building for dimension ' num2str(i) ' and output position ' num2str(q) '\n']);
%         T(i,:,q,:) = [zeros(d1,(q-1)*dW), k_r(:,:,i), zeros(d1, length_input - (q-1)*dW - size(k_r,2))];
%     end
% end

% T can be studied as an operator now