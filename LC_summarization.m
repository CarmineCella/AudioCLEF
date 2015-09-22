function [Fs, labels_s, entries_s] = LC_summarization (F, labels, entries, params)
    Fs = [];
    labels_s = [];
    entries_s = [];

    switch (params.type)
        case 'mean_std'
            disp ('summarizing by mean and std deviation...');  
            for i = 1 : max (entries)
                g = F(:, entries == i);
                gi = labels(entries == i);
                tt = [mean(g, 2) std(g, 0, 2)];
                Fs = [Fs tt(:)];
                assert(length(unique(gi))==1);
                labels_s = [labels_s unique(gi)];
                entries_s = [entries_s i];
            end
        case 'max'
            disp ('summarizing by max...');  
            for i = 1 : max (entries)
                g = F(:, entries == i); 
                gi = labels(entries == i);
                w = sum (g);
                [~,sortIndex] = sort (w(:), 'descend');
                a = params.components;
                q = [];
                if a > length (sortIndex)
                    for j = 1 : a;
                        q = [q g];
                    end
                else
                    maxIndex = sortIndex (1 : a); %# Get a linear index into A of the 5 largest values
                    q = g(:, maxIndex);            
                end
                
                Fs = [Fs q(:)];      
                labels_s = [labels_s unique(gi)]; % always averaging labels
                entries_s = [entries_s i];
            end
%         case 'diff_map'
%             disp ('summarizing by diffusion maps...');  
%             for i = 1 : max (entries)
%                 g = F(:, entries == i);
%                 gi = labels(entries == i);
%                 if size (g, 2) == 1
%                     Fs = [Fs zeros(params.components, 1)];
%                     labels_s = [labels_s mean(gi)]; % always averaging labels                   
%                 else                
%                     diff_mat = difference_matrix (g, 1);
% 
%                     [variance] = estimate_variance (diff_mat, g, 1);
%                     [~, ~, L] = make_laplacian (diff_mat, variance, .5);
%                     [u,d] = eig(L);
%                     [~, ma]= sort (diag(d), 'descend');
%                     U = u(:, ma);
% 
% %                     principal_eig = U(:,[2:3]);
% % 
% %                     yhisto = hist (principal_eig(:), params.components);
% 
%                     tt = L(:);
%                     size (tt)
%                     Fs = [Fs tt(1:params.components)];
%                     labels_s = [labels_s mean(gi)]; % always averaging labels
%                 end
%             end
        case 'nnmf'
            disp ('summarizing by nnmf...');  
            for i = 1 : max (entries)
                g = F(:, entries == i);
                gi = labels(entries == i);                           
                w = nnmf (g, params.components);
                a = params.components;
                q = [];
                if a >= length (w)
                    for j = 1 : a;
                        q = [q w];
                    end
                else
                    q = w(:);
                end
                
                Fs = [Fs q(:)];      
                labels_s = [labels_s mean(gi)]; % always averaging labels
                entries_s = [entries_s i];
            end         
        case 'none'
            Fs = F;
            labels_s = labels;
            entries_s = entries;
            
        otherwise
            error ('LifeClef2015 error: invalid summarization');
    end
end
