# LifeClef (birds task)

This project contains experimental MATLAB code for the birds task of the [LifeClef project](http://www.imageclef.org/lifeclef/2015).

Dependencies:

[ScatNetLight](https://github.com/edouardoyallon/ScatNetLight/releases) by E. Oyallon

[scattering.m](https://github.com/lostanlen/scattering.m) by V. Lostanlen

[MFCC library](http://labrosa.ee.columbia.edu/matlab/rastamat/) by D. Ellis


The main file is LC_main.m; it invokes the computation function (called LC_batch) which takes several parameters and gives back the computed features, the labels, the classification results and other things.

Currently, the main file runs a batch of classifications depending on several variables; if you want to run a single classification use LC_batch directly after setting the required parameters.

For more information contact carmine.emanuele.cella@ens.fr

(c) 2015 carmine e. cella
