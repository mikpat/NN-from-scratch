function [y, yy] = last_forward_step(prev_layer, weights)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

if size(weights,1) ~= size(prev_layer,2)+1
    error('Size of weights matrix does not match dimensions of prev_layer!')
end

yy = [ones(size(prev_layer,1),1) prev_layer] * weights;
y = exp(yy)./sum(exp(yy),2); 

end
