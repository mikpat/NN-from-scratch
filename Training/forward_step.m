function [h, z] = forward_step(prev_layer, weights)

if size(weights,1) ~= size(prev_layer,2)+1
   error('Size of weights matrix does not match dimensions of prev_layer!')
end

z = [ones(size(prev_layer,1),1) prev_layer] * weights;
h = (z > 0).*z; 

end

