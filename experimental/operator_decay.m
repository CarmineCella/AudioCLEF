k = x;
k_r = reshape (k, 40, 5, 80);
k_p = pad_signal (k_r, [40 250 80], 'zero', 0);
K = fft (k_p, [], 2);

clear t;
for i = 1 : 250
   t(i, :) = eig(squeeze (K(:, i, :))' * squeeze (K(:, i, :)));
end
t = log (abs (t(:)));
plot (sort (t, 'descend'))


 