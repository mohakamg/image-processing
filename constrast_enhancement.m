%contrast enhancement
close all
clc, clear;
L = 256;
r = rgb2gray(imread('paris_879.jpg'));

unique_pixel_intensities = double(unique(r));

[m,n] = size(r);
pixel_count_total = m*n;

for i=1:length(unique_pixel_intensities)
   unique_indeces{i} = find(r == unique_pixel_intensities(i));
   
   P_r(i) = length(unique_indeces{i})/pixel_count_total;
end
