function correct_per = test_NN( test_data, test_labels, w )

classification_NN = zeros(size(test_labels, 1), 1);

for i=1:size(test_labels, 1)
    % Current observation
    X = test_data(:, i)';
    
    % Forward propagation through all network
    [~, z] = forwardprop_all_layers(X, w);
    
    % Transform last layer activation to output
    Y_pred = exp(z{end})./sum(exp(z{end}),2);
    [~, Y_max_index] = max(Y_pred);
    
    classification_NN(i) = (Y_max_index-1);

end

correct_per = int8(100*sum(test_labels == classification_NN)/size(test_labels, 1));

end

