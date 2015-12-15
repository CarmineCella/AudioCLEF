close all
load '../torch/l3_weights.mat';

d1 = 96;
d2 = 320;
families = 2;
alpha = .5;
k = x';
k_r = reshape (k, d1, 5, d2);
%%
figure
for i = 1 : d2
    subplot (d2/10, 10, i)
    v = fft2 (squeeze (k_r (:, :, i)), d1, 50);
    f = abs (v );
    imagesc (f((1:end/2), :))
end

%%
k_p = pad_signal (k_r, [d1 50 d2], 'zero', 0);
K = fft (k_p, [], 2);
aK = abs (K);
maK = squeeze (max (aK, [], 2))';
dif = squareform (pdist (maK));
%%
[yhisto, xhisto] = hist (dif(:), d2);
var = (sum (yhisto .* xhisto / sum (yhisto))) * .6;
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
%%
f1 = maK (idx == 1, :);
f2 = maK (idx == 2, :);

figure
subplot (211)
imagesc (f1)
subplot (212)
imagesc (f2)
%%
figure
for i = 1 : d2
    subplot (d2/10, 10, i)
    imagesc (abs (K(:,(1:end/2),i)))
end
%%
clear t;
for i = 1 : 50
    t(i, :) = eig(squeeze (K(:, :, i))' * squeeze (K(:, :, i)));
end
t = abs (t (:)); %log (abs (t(:)));
figure
plot (sort (t, 'descend'))
%% Build the operator T

% Assume our input signals are represented as matrices s, such that s_{j,r}
% represents the value of coordinate j at time r
% Assume that our output signals are represented as matrices Ts,  such that
% (Ts)_{i,q} is the value of coordinate i at time q
% T is represented by a tensor T_{i,j,q,r} such that:
% (Ts)_{i,q} = \sum_{j} \sum_{r} T_{i,j,q,r} s_{j,r}
% T can be constructed from the filters in k (of size K*(KW)*K') in the
% following manner:
length_output = 46;
length_input = 50;
dW = 1;

T = zeros(d2,d1,length_output,length_input); % warning, can become very large

for i = 1:d2
    for q = 1:length_output
       % fprintf(['Building for dimension ' num2str(i) ' and output position ' num2str(q) '\n']);
        T(i,:,q,:) = [zeros(d1,(q-1)*dW), k(:,:,i), zeros(d1, length_input - (q-1)*dW - size(k,2))];
    end
end

% T can be studied as an operator now