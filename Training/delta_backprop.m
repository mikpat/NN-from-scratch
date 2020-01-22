function delta = delta_backprop( delta_next, w_next, z)
% Returns gradient of current layer units
%   based on next layer gradients 'delta_next', next layer weights 'w_next'
%   and weight sum of all inputs to current layer 'z'

delta = w_next*delta_next;
% Inlcude bias unit
one_vec = ones(size(delta,2),1);
z = [one_vec z];
% Multiplication with derivitive of the ReLU. ReLU'=0 for z<0 && ReLU'=1 for z>0
delta(z<0)=0;

end

