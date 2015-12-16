% operator study
%

close all

%% load data
load 'weights/exp2_l1_weights_41.mat';

%% params and reshape
d1 = 40;
sup = 41;
d2 = 40;
space = 250;
families = 6;
alpha = .5;
var_ratio = .8;
denoise_spatial = 2.4;
denoise = 2.4;

k_r = reshape (x, d1, sup, d2);

%% plot spatial filters
figure
for i = 1 : d2
    subplot (d2/10, 10, i)
    s =(k_r(:, :, i));
    s(abs(s)<denoise_spatial*mean(abs(s(:)))) = 0;
    imagesc (s)
end

%% calculate 2D fft by padding
k_p = pad_signal (k_r, [d1 space d2], 'zero', 0);
K = fft2 (k_p);
aK = abs (K);
aK_th = zeros(size(aK));
figure
for i = 1 : d2
    subplot (d2/10, 10, i)
    s = aK(:, :, i);
    s(abs(s)<denoise*mean(abs(s(:)))) = 0;
    aK_th(:,:,i) =s;
    %a = ifft (log( s));
    %a((16:end), :) = zeros (d2-15, size (a, 2));
    %e = abs (fft (a));
    imagesc (fftshift (s))
end
%%
% sum_f = zeros (size (aK, 1), size(aK, 2));
% for i = 1 : d2
%     sum_f = sum_f + aK(:, :, i);
% end
% imagesc (fftshift (sum_f))

%% compute distance matrix
dist = zeros (d2, d2);
for i = 1 : d2
   for j = 1 : d2
       d = ( (aK (:, :, i)- aK (:, :, j))).^2;
      dist (i,j) = sqrt (sum (sum (d)));
   end
end
figure
imagesc (dist)

%% embed matrix and make clusters
[yhisto, xhisto] = hist (dist(:), d2);
var = (sum (yhisto .* xhisto / sum (yhisto))) * var_ratio;
A = exp (-(dist.^2)/(var^2));
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
principal_eig = [U(:,2:3)];
[idx1, C1] = kmeans (principal_eig, families);
 
figure
scatter (principal_eig(:,1), principal_eig(:, 2),[], idx1)
%% show clusters
for i = 1 : families
    I = find (idx1 == i);
    figure
    
    my_size = ceil(sqrt(numel(I)));
    for j = 1 : numel (I)
        subplot (my_size, my_size, j)
        imagesc (k_r(:, :, I(j)))
    end
    
end
%% calculate spectrum of the operator
clear t;
for i = 1 : size (K, 2);
    t(i, :) = eig (squeeze (K(:, i, :))' * squeeze (K(:, i, :)));
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
