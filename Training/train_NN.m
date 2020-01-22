function [w, loss] = train_NN(X, Y, architecture, n_inputs,...
                                n_outputs, l_rate, epochs, ...
                                drop_size, batch_size, dev_perc)

% Function train_NN: trains a neural network with flexible number 
%                    of layers and hidden units

    % NN architecture
    n_h_layers = size(architecture, 2);

    % Init random weights
    w = random_weights(n_h_layers, architecture, n_outputs, n_inputs);

    % If error accours then found_NN = 0 and it will search again
    found_NN = 0;

    % Store history of the cross entropy loss function
    L = zeros(size(Y,1)/batch_size,1);

    loss_epochs_train = zeros(epochs, 1);
    loss_epochs_dev = zeros(epochs, 1);

    batches_starting_indecies = 1:batch_size:size(X,2);
    
    devset_idx = randperm(size(X,2), round(dev_perc*size(X,2)));
    devset_X = X(:,devset_idx)';
    devset_Y = Y(devset_idx);
    
    train_idx = ones(size(X,2),1);
    train_idx(devset_idx) = 0;
    X = X(:,logical(train_idx));  
    Y = Y(logical(train_idx));
    
    for epoch=1:epochs
        shuffle = randperm(size(X,2));
        X = X(:,shuffle);  
        Y = Y(shuffle);

        for i=1:(size(Y,1)/batch_size)
            % Current batch
            X_batch = X(:, batches_starting_indecies(i):batches_starting_indecies(i)+(batch_size-1))';
            Y_batch = Y(batches_starting_indecies(i):batches_starting_indecies(i)+(batch_size-1)); 

            [h, z] = forwardprop_all_layers_dropout(X_batch, w, drop_size);
            % Transform last layer activation to output using softmax
            Y_pred = exp(z{end})./sum(exp(z{end}),2);

            % t -> vector containing correct labels as 1 and others as 0
            t = zeros(size(Y_pred+1));
            for j = 1:size(Y_pred)
                t(j, Y_batch(j)+1) = 1;
            end
            % Cross entropy loss function
            L(i) = -sum(log(Y_pred(t==1)));

            % Backpropagate through all layers to obtain gradients
            delta = backprop_all_layers(Y_pred, t, n_h_layers, w, z);  

            % New weights
            [w, error_NaN] = new_weights(w, delta, X_batch, z, l_rate);

        end

        % Evaluate dev set error
        [h, z] = forwardprop_all_layers(devset_X, w);
        Y_pred_dev = exp(z{end})./sum(exp(z{end}),2);
        t = zeros(size(Y_pred_dev+1));
        for j = 1:size(Y_pred_dev)
            t(j, devset_Y(j)+1) = 1;
        end
        loss_epochs_dev(epoch) = -sum(log(Y_pred_dev(t==1)))/size(Y_pred_dev,1);
        
        
        loss_epochs_train(epoch) = sum(L)/(size(L,1)*batch_size);
    end

    loss.loss_train = loss_epochs_train;
    loss.loss_dev = loss_epochs_dev;
end

    
    