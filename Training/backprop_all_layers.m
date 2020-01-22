function delta_h = backprop_all_layers( Y_pred, t, n_h_layers, w, z )
    
    delta_output =  (Y_pred - t)';
    delta_h{n_h_layers+1} =  delta_output;
    for i = (n_h_layers):-1:1
        if i==(n_h_layers)
            delta_h{i} = delta_backprop(delta_output, w{i+1}, z{i});
        elseif i == 1
            % Don't include bias unit in backprop
            delta_current = delta_h{i+1}(2:end,:);
            delta_h{i} = delta_backprop(delta_current, w{i+1}, z{i});
        else
            % Don't include bias unit in backprop
            delta_current = delta_h{i+1}(2:end,:);
            delta_h{i} = delta_backprop(delta_current, w{i+1}, z{i});
        end
    end

end

