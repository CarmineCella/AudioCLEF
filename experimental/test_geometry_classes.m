% tests on class geometry
%

addpath (genpath ('../HSC'));

params = struct ('filename', '', ...
    'T', 2^11, ...
    'renormalize', 0, ...
    'Lp_norm', 2, ...
    'variance_scales', [1 1], ...
    'plot_scattering', 1, ...
    'plot_results', 1, ...
    'plot_summaries', 1 );

order = length (params.variance_scales); % NB: the size of the variance_scales vector
                                         % determines the order of the network

%% Sum all vectors within each class
F_sum = AC_summarization(F, labels, entries, summarization_params);
F_byclass = sum(reshape(F_sum, 80, 15, 15), 3);

sc_lin = F_byclass;
sc_log = log1p (1e-12 * sc_lin);
imagesc(sc_log)
%% decompose
s = size (sc_log);
indexes = zeros (1, s (2));
for i = 1 : s(2)
    indexes (1, i) = i;
end

paths = decompose_path (sc_log, sc_lin, indexes, params, order, 'left');


%% plot results
close all

if (params.plot_summaries == 1)
    plot_summaries (paths, sc_log, params);
end

if (params.plot_results == 1)
    for h = 1 : order
        plot_order (paths, h);
    end
end

if (params.plot_scattering == 1)
    figure
    set(gcf,'numbertitle','off','name', 'Original signal')
    subplot (2, 1, 1)
    imagesc (sc_log);
    title ('Log joint scattering vectors (1, 2)');
    subplot (2, 1, 2)
    imagesc (sc_lin);
    title ('Lin joint scattering vectors (1, 2)');
    
end