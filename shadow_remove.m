function [L_hat] = test(name)

pkg load image

I_crop1 = imread(name);
    %I_crop1=imresize(I_crop3,0.5);
    n = 2;
Ig1=(double(I_crop1)/255).^4;
L=(Ig1.^n)./ (flipud(cumtrapz(flipud(Ig1.^n))));
L_hat = uint8(255*(L).^(1/4));
