function [F_r, labels_r, entries_r] = LC_distribution_eq (F, labels, entries, Nclass)
disp ('equalizing distribution of samples...');
maxel = min (hist (labels, Nclass));

fprintf ('\tmax elements per class = %d\n', maxel);
F_r = [];
entries_r = [];
labels_r = [];

for i = 1 : Nclass
    F_inclass = F(:, labels == i);
    entries_inclass = entries(labels == i);
    labels_inclass = labels(labels==i);
    
    F_r = [F_r F_inclass(:, 1:maxel)];
    entries_r = [entries_r entries_inclass(:, 1:maxel)];
    labels_r = [labels_r labels_inclass(:, 1:maxel)];
end

end