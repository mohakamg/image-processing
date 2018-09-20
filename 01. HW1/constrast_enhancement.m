%contrast enhancement
close all
clc, clear;
L = 256;
r = imread('Testimage1.tif');
img = r;

[m,n] = size(r);
pixel_count_total = m*n;
img = double(reshape(img, [1, pixel_count_total]));
unique_pixel_intensities = double(unique(img));

for i=1:length(unique_pixel_intensities)
   unique_indeces{i} = find(img == unique_pixel_intensities(i));
   P_r(i) = length(unique_indeces{i})/pixel_count_total;
   img(unique_indeces{i}) = 255*sum(P_r(1:i));
end

stem(unique_pixel_intensities, P_r)
title('histogram');
img = reshape(img, [m, n]);

img = correct_negatives(img);
figure
%imshow([img, histeq(r)]);
imshow([img, r]);

int_filter = 1/9*[1 1 1; 1 1 1; 1 1 1];

function S = correct_negatives(r) %corrects negative pixels
    [m, n] = size(r);
    r = double(r);
    r_reshaped = reshape(r, [1,m*n]);
    min_r = min(r_reshaped);
    max_r = max(r_reshaped);
    S=uint8(255.*(r- min_r)./(max_r-min_r));
end

function S = neg_map(L, r)
    S=uint8((L-1)-r);
end