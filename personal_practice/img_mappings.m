close all
clc, clear;
r = rgb2gray(imread('paris_879.jpg'));

L = 256;
imshow(neg_map(L, r));
pause
C= 1;
imshow(correct_negatives(log_map(C, r)));
pause
C= 1;
alpha = 0.8
imshow(correct_negatives(exp_map(C,alpha, r)));
pause
close all

%negative mapping
function S = neg_map(L, r)
    S=uint8((L-1)-r);
end

function S = log_map(C, r) %log mapping
    S=C*log(1 + double(r));
end

function S = exp_map(C, alpha, r) %exponential mapping
    %S=C.*exp(double(r));
    r = double(r);
    S=C.*r.^(alpha);
end