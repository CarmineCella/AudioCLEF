function [moys,stddevs, x_out] = LC_standardization (x_in)
    epsilon = eps (); %1e-5;

% to check!!
%     moys = mean(x_in, 1) ; % fixme
%     x_out = bsxfun(@minus, x_in, moys);
%     stddevs = sqrt(sum(x_out.^2,1));
%     x_out = bsxfun(@rdivide,x_out,stddevs+epsilon);
%     
     moys = mean(x_in,2);
     stddevs = std(x_in,1,2);
     numFrames = size (x_in,2);
     x_out = (x_in - repmat (moys,1,numFrames))./repmat(stddevs,1,numFrames);
end

