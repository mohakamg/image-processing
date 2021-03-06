%contrast enhancement
close all
clc, clear;
L = 256;
I_original = imread('Testimage1.tif');
%mistake; this is not the average
%int_filter = 1/9*[1 1 1 1 1 ; 1 1 1 1 1 ; 1 1 1 1 1];
%% integration - blur
int_filter = 1/25*[1 1 1 1 1 ; 1 1 1 1 1 ; 1 1 1 1 1];
I_blur = conv2(I_original,int_filter,'same');
I_blur = uint8(I_blur);

%imshow(I_blur);
%% blur /integration
I_bin = I_blur;
% I_bin(I_bin < 235) = 0; %threshold to practially binarize the img
%% correcting negatives

I_blur = correct_negatives(I_blur);
I_blur(I_blur<75)=0; %threshold to practially binarize the img
I_blur(I_blur>=55)=255; %threshold to practially binarize the img

%% remove spades / nums
I_rem = remove_spades_numbers(I_blur);
%% integration - blur
int_filter = 1/9*[1 1 1; 1 1 1; 1 1 1];
I_blur_rem = conv2(I_rem,int_filter,'same');
I_blur_rem = conv2(I_blur_rem ,int_filter,'same');
I_blur_rem = uint8(I_blur_rem);

%% 1st derivative
% x_1st_deriv = [0 1 0; 1 -4 1; 0 1 0];
% y_1st_deriv = [-1 0 1; -2 0 2; -1 0 1];
x_1st_deriv = [1 1; -1 -1];
y_1st_deriv = [1 -1; 1 -1];
I_1s_tder_x = conv2(I_blur_rem, x_1st_deriv ,'same');
I_1s_tder_y = conv2(I_blur_rem, y_1st_deriv ,'same');
%% 2nd derivative
sec_deriv = [0 1 0; 1 -4 1; 0 1 0];
I_sec_deriv = conv2(I_blur_rem, sec_deriv,'same');
%I_sec_deriv = conv2(I_sec_deriv, sec_deriv,'same');
figure
imshow([I_1s_tder_x, I_1s_tder_y, I_sec_deriv]);
%% display figs
figure
I_sec_deriv = contrast_enhancement(I_sec_deriv);
I_sec_deriv(I_sec_deriv > 200) = 255;
I_sec_deriv(I_sec_deriv <= 200) = 0;
% imshow([I_original, I_blur, I_blur_contr_enhanced, I_bin]);
imshow([I_original, I_blur, I_bin,I_rem, I_sec_deriv]);
title('original VS blur 5x5 VS i blur constrast enhanced vs binarized');

function I_rem = remove_spades_numbers(I)
    [m,n] = size(I);
    counter_black = 0;
    counter_white = 0;
    change_from_white_to_black = 0;
    change_from_black_to_white = 0;
    l = m*n;
    
    for i=2:l
        if I(i-1) >= 100 && I(i) < 100
            change_from_white_to_black = 1;
            change_from_black_to_white = 0;
        elseif I(i-1) < 100 && I(i) >= 100
            change_from_white_to_black = 0;
            change_from_black_to_white  = 1;
        end
        % gets rid of black shapes within the card
        if I(i) < 100 && change_from_white_to_black
            counter_black = counter_black  + 1;
            %black_indeces = i;
        end
        
        if change_from_black_to_white
            if counter_black > 0 && counter_black < 100
                I(i-counter_black:i) = 255;
            end
            counter_black = 0;
        end
        
    end
    
    % gets rid of remaining white noise
    for i=2:l
        if I(i-1) >= 100 && I(i) < 100
            change_from_white_to_black = 1;
            change_from_black_to_white = 0;
        elseif I(i-1) < 100 && I(i) >= 100
            change_from_white_to_black = 0;
            change_from_black_to_white  = 1;
        end
    
        if I(i) > 100 && change_from_black_to_white
            counter_white = counter_white  + 1;
        end

        if change_from_white_to_black
            if counter_white > 0 && counter_white < 10
                I(i-counter_white:i) = 0;
            end
            counter_white = 0;
        end

    end
    I_rem = I;
end

function [img, unique_pixel_intensities, P_r] = contrast_enhancement(img)
    [m,n] = size(img);
    pixel_count_total = m*n;
    img = double(reshape(img, [1, pixel_count_total]));
    unique_pixel_intensities = double(unique(img));
    for i=1:length(unique_pixel_intensities)
       unique_indeces{i} = find(img == unique_pixel_intensities(i));
       P_r(i) = length(unique_indeces{i})/pixel_count_total;
       img(unique_indeces{i}) = 255*sum(P_r(1:i));
    end
    img = reshape(img, [m, n]);
end
function S = correct_negatives(r) %corrects negative pixels

    index_negatives = find(r < 0);
    if ~isempty(index_negatives)
        % index_negatives
        [m, n] = size(r);
        r = double(r);
        r_reshaped = reshape(r, [1,m*n]);
        min_r = min(r_reshaped);
        max_r = max(r_reshaped);
        S=uint8(255.*(r- min_r)./(max_r-min_r));
    else
        S = r;
    end
end

function S = neg_map(L, r)
    S=uint8((L-1)-r);
end