% Test file for average-log coefficients

addpath ('../../../libs/mfcc');
addpath ('../lib')

FFTsize = 2048;
FFTolap = FFTsize / 2;
nbands = 40;
ncoeff = 13;
alpha = 3.5;

%%
[s, sr] = audioread ('../../../datasets/various_data/Bach_preludeu.wav');

%% 
[G, ff1] = AC_AverageLogCoeff (s, FFTsize, FFTolap, nbands, ncoeff, alpha);

%%
  [gg, ff] = melfcc (s, sr,'numcep', ncoeff, 'nbands', nbands, ...
    'wintime', FFTsize / sr, 'hoptime', FFTsize / sr);
%%

figure
subplot (4, 1, 1)
imagesc ((ff1))
title ('AL spectrum');
subplot (4, 1, 2)
imagesc ((G))
title ('AL coeff');
subplot (4, 1, 3)
imagesc ((ff))
title ('Mel spectrum');
subplot (4, 1, 4)
imagesc ((gg))
title ('Mel coeff')


%%
% P = [];
% for j = 1 : nBands
%     lambda = G(j,:);
%     
%     t = 512;
%     p = spectrogram (lambda, t);
%     a = abs (p);
%     P = [P a(:)];
% end
% 
% ff = vertcat (G, P*G);