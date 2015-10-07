# AudioCLEF

AudioCLEF is an experimental MATLAB framework for audio classification. It Currently supports several audio features, among which MFCC and scattering, and two classifiers (SVM and RandomForests). It is possible to change data before the final classification by means of normalization, standardization and supervised dimensionality reduction (OLS). 

It can be used with any dataset, the only constraint is that each class should be in a separate folder. For example:

- class 1
	* file1.wav
	* file2.wav
	* file3.wav
	
- class 2
	* file1.wav
	* file2.wav


Current dependencies are:

[ScatNetLight](https://github.com/edouardoyallon/ScatNetLight/releases) by E. Oyallon

[scattering.m](https://github.com/lostanlen/scattering.m) by V. Lostanlen

[MFCC library](http://labrosa.ee.columbia.edu/matlab/rastamat/) by D. Ellis


The main file invokes the computation function (called AC_batch) which takes several parameters and gives back the computed features, the labels, the classification results and other things.

It can run a batch of classifications depending on several variables; if you want to run a single classification use LC_batch directly after setting the required parameters.

For more information contact carmine.emanuele.cella@ens.fr

(c) 2015 carmine e. cella
