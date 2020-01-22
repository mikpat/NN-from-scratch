function w = random_weights( n_h_layers, n_h_units, n_outputs, n_inputs )

variance = sqrt(2/n_inputs);
for i=1:n_h_layers
    if(i==1)
        w{i} = variance*randn(n_inputs+1, n_h_units(i));
    else
        w{i} = variance*randn(n_h_units(i-1)+1, n_h_units(i));
    end
    if i == n_h_layers
        w{i+1} = variance*randn(n_h_units(i)+1, n_outputs);
    end
end

end

