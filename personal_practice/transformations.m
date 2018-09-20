%geometric transformations
close all
clc, clear;

rotation=@(angle_dg)[cosd(angle_dg),-sind(angle_dg),0;...
                    sind(angle_dg), cosd(angle_dg),0;...
                                                0,0,1];

rotation=@(angle_dg)[cosd(angle_dg),-sind(angle_dg),0;...
                    sind(angle_dg), cosd(angle_dg),0;...
                                                0,0,1];
                                            
translation=@(t_x,t_y)[1,0,t_x; 0, 1,0;0,0,t_y];
shear_vert=@(s_0)[1,s_0,0; 0, 1,0;0,0,0];
shear_hor=@(s_0)[1,0,0; s_0, 1,0;0,0,0];

L = 256;

img = rgb2gray(imread('paris_879.jpg'));
r = img;
[m,n] = size(img);
x_arr = 1:n;
y_arr = 1:m;
T = rotation(-30);

T_boundaries = T*[n m 1]';
new_img = zeros(round(T_boundaries(2)),round(T_boundaries(1)));
for x=1:n
    for y=1:m
        rotated = T*[x y 1]';
        x_rot = round(rotated(1)); 
        y_rot = round(rotated(2));
        if ~x_rot
            x_rot = 1;
        elseif ~y_rot
            y_rot = 1;
        end
        new_img(y_rot,x_rot) = r(y,x);
        
    end
end

imshow(new_img);

imshow(uint8(r));