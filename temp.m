%        
%         majority_labels = zeros (size (y_hat, 1), 1); % tmp
%         for i = 1 : size (y_hat, 1)
%             majority_labels(i) = str2double (y_hat{i});
%         end


% 
% TP_matrix = cumsum(boolean_matrix,2);
% avgTP = mean(TP_matrix,1);
% precision = avgTP ./ (1:Nclass);
% 
% diffrecall = mean(boolean_matrix,1);
% APs = precision .* diffrecall;
% map = sum(APs);

%assert(all(predicted_matrix(:,1) == majority_labels));
%perf = classperf (test_labels, predicted_labels); % FIXME!!



%     mapk = zeros (Nclass, 1);
%     for k = 1:Nclass
%         mapk(k) = LC_MAP_at_K (test_labels, predicted_matrix, k);
%     end
%     map = mean (mapk);