close all
load l1.mat

coeff = 5;
width = 5;

l1 = x.';

figure
imagesc (l1)

xc = zeros(size(l1));
for i = 1 : size(xc,2)
  xc(:,i)= abs(fft(l1(:,i)));
end

%xc=xc(1:end/2, :);
figure
imagesc (xc)
xc = sort(xc)

%%

xcf = zeros(size(xc));
for i = 1 : size(xc,2)
    d = xc(:, i);
    %c = conv (xc(:, i), hann (width));
    lifter = zeros(1, length(d)-coeff);
    cep = ifft (log(d));
    cep(coeff+1:end) = lifter;
    specenv = abs(fft(cep));
    
    xcf(:, i) = specenv(1:size(xc,1));
end


figure
imagesc (xcf)

%%

x = repmat (1:size(xcf,2), size(xcf,1),1);
x= x(:);
y = repmat (1:size(xcf,1), size(xcf,2),1)';
y=y(:);
z = reshape(xcf, numel(xcf),1);

figure
hold on
for i=1:size(xcf,2)
    I =(i-1)*size(xcf,1) +1 : i*size(xcf,1);
    plot3(x(I), y(I), z(I))
end


