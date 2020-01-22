function [h, z] = forwardprop_all_layers_dropout( X, w, drop_size)
%Forward propagation through all network
%   stores weighted sum of inputs as z
%   stores activations as h(first h cell is input data)
    
for j = 1:size(w,2)
    
    if j == 1
        [h{j}, z{j}] = forward_step(X, w{j});
    else
        [h{j}, z{j}] = forward_step(h{j-1}, w{j});
    end
    
    if j<size(w,2)
    	dropout_indicies = randi(size(h{j}, 2), drop_size(j), 1);
        h{j}(:,dropout_indicies) = 0;
        z{j}(:,dropout_indicies) = 0;
    end
end

end

