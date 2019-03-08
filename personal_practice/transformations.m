%geometric transformations
close all
clc, clear;

rotation=@(angle_dg)[cosd(angle_dg),sind(angle_dg),0;...
                    -sind(angle_dg), cosd(angle_dg),0;...
                                                0,0,1];

% rotation=@(angle_dg)[cosd(angle_dg),-sind(angle_dg),0;...
%                     sind(angle_dg), cosd(angle_dg),0;...
%                                                 0,0,1];
                                            
translation=@(t_x,t_y)[1,0,t_x; 0, 1,0;0,0,t_y];
shear_vert=@(s_0)[1,s_0,0; 0, 1,0;0,0,0];
shear_hor=@(s_0)[1,0,0; s_0, 1,0;0,0,0];

L = 256;

%img = rgb2gray(imread('paris_879.jpg'));
%img = uint8(zeros(2,2));

% [m,n] = size(img);
% rotated = 0;
% if m > n
%     img = imrotate(img,90);
%     img = imrotate(img,90);
%     rotated = 1;
% end


angle_dg = -45;
T = rotation(angle_dg);

img = uint8([255   0 255   0 255;...
               0 255   0 255   0;...
             255   0 255   0 255;...
               0 255   0 255   0;...
             255   0 255   0 255]);
         
[m,n] = size(img);
r = img;

T_boundaries = T*[n m 1]';
%new_img = zeros(round(T_boundaries(2)),round(T_boundaries(1)));
new_img = uint8(zeros(15,15));

new_coord_x_y = [0,0];
for y=1:m
    for x=1:n
        rotated = [x y 1]*T;
        %rotated = T*[x y];
%         x_rot = round(rotated(1)+5*cosd(angle_dg));
%         y_rot = round(rotated(2)-5*sind(angle_dg));
        x_rot = round(rotated(1));
        y_rot = round(rotated(2));        
%         x_rot = round(x*cosd(-30)); 
%         y_rot = round(y*sind(-30)); 
        
%         if x_rot < 0
%             x_rot = -x_rot;
%         end
%         if y_rot < 0 
%             y_rot = -y_rot;
%         end
        new_coord_x_y(x,:) = [x_rot y_rot];
     %   new_img(new_coord_x_y(x,1),new_coord_x_y(x,2)) = r(y,x);        
    end
end

index_x_less_than_zero = find(new_coord_x_y(:,1) <= 0);
%x_less_than_zero = new_coord_x_y(:,1) < 0
index_y_less_than_zero = find(new_coord_x_y(:,2) <= 0);

if length(index_x_less_than_zero) > 0
    new_coord_x_y(index_x_less_than_zero,1) =...
        1 - new_coord_x_y(index_x_less_than_zero,1);
end

if length(index_y_less_than_zero) > 0
    new_coord_x_y(index_y_less_than_zero,2) =...
        1 - new_coord_x_y(index_y_less_than_zero,2);
end

new_img(new_coord_x_y(:,2),new_coord_x_y(:,1)) = uint8(r(1:y,1:x));
%new_img(1:y,1:x) = r(new_coord_x_y(:,2),new_coord_x_y(:,1));
figure
subplot(1,2,1)
imshow(new_img);
subplot(1,2,2)
imshow(r);

%imshow(uint8(r));