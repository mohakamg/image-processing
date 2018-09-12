%contrast enhancement
close all
clc, clear;
L = 256;
r = rgb2gray(imread('paris_879.jpg'));
img = r;

%imshow(img);

[m,n] = size(r);
pixel_count_total = m*n;
img = double(reshape(img, [1, pixel_count_total]));
unique_pixel_intensities = double(unique(img));

for i=1:length(unique_pixel_intensities)
   unique_indeces{i} = find(img == unique_pixel_intensities(i));
   
   P_r(i) = length(unique_indeces{i})/pixel_count_total;
   %cummulative sum (
   img(unique_indeces{i}) = 255*sum(P_r(1:i));
   
  % img(unique_indeces{i}) = round(sum(img(unique_indeces{i})));
end
%subplot(2,1,1)
stem(unique_pixel_intensities, P_r)
title('histogram');
%subplot(2,1,2)
img = reshape(img, [m, n]);

img = correct_negatives(img);
imshow([r, img, histeq(r)]);