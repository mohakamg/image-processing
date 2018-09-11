function S = correct_negatives(r) %corrects negative pixels
    r = double(r);
    S=uint8(255.*(r-min(r))./(max(r)-min(r)));
end