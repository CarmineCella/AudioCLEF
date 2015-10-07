function [alcoeff, alspec] = AC_AverageLogCoeff (signal, FFTsize, FFTolap, nbands, ncoeff, alpha)
    FFTsize2 = FFTsize / 2;
    S = spectrogram (signal, FFTsize, FFTolap);
    M =  (abs (S));

    H = [0:1/nbands:1];

    %% building exponential scale
    c1 = 1. / (1.0 - exp (alpha));
    lspace = c1 * (1.0 - exp (H .* alpha));
    lspace = lspace .* FFTsize2;
    lspace = floor (lspace(2:end));

    %% averaging on exponential scale
    alspec = [];
    for j = 1 : size (M, 2)
        F = [];
        for i = 1 : (length (lspace)-1)
            chunk = (M(lspace(i):lspace(i+1), j));
            w = bartlett (lspace(i+1) - lspace(i) + 1);
            f =  mean (chunk .* w);
            F = [F f];
        end
        alspec = [alspec F'];
    end

    %% transforming along frequency
    alcoeff = [];
    for j = 1 : size (alspec, 2);
        lambda = log (alspec(:, j));

        p = dct (lambda, 32);
        alcoeff = [alcoeff p(:)];
    end

    alcoeff = alcoeff(2:ncoeff, :);
end
