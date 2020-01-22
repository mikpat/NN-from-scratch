clear all
close all

addpath ./Training

%% Load data

data_mnist = load("MNIST.mat");
test_data = data_mnist.test_data;
test_labels_one_hot = data_mnist.test_label;
train_data = data_mnist.train_data;
train_labels_one_hot = data_mnist.train_label;

%% Transform one hot encodings to ints for train and test sets

train_labels = zeros(size(train_labels_one_hot,1),1);
for i=1:size(train_labels_one_hot,1)
    train_labels(i) = find(train_labels_one_hot(i,:)==1)-1;
end

test_labels = zeros(size(test_labels_one_hot,1),1);
for i=1:size(test_labels_one_hot,1)
    test_labels(i) = find(test_labels_one_hot(i,:)==1)-1;
end

% Dataset params
n_inputs = 784;
n_outputs = 10;


%% Define hyperparameters for a NN

architecture = [300, 600];
l_rate=0.0002;
batch_size = 100;
epochs = 40;
drop_size = [60, 120];
dev_perc = 0.2;


%% Train NN

[NN_weights, loss] = train_NN(train_data', train_labels, architecture,...
                                n_inputs, n_outputs, l_rate, epochs, ...
                                drop_size, batch_size, dev_perc);

                            
                            
%% Predict and test

[correct_per, predictions] = test_NN(test_data', test_labels, NN_weights);


%% Results

[precision, recall, F1] = classification_measures(test_labels, predictions);
confusion_matrix(test_labels, predictions)

fprintf('Evaluation\n\n');
fprintf('Accuracy: %0.3f%%\n\n', correct_per);
fprintf('Precision: %0.3f%%\n\n', precision*100);
fprintf('Recall: %0.3f%%\n\n', recall*100);
fprintf('F1 score: %0.3f%%\n\n', F1*100);

figure
hold on
plot(1:size(loss.loss_train,1), loss.loss_train)
plot(1:size(loss.loss_dev,1), loss.loss_dev)
legend("Training loss", "Development loss")
title("Log-loss functions per observation")
xlabel("Epochs")
ylabel("Log-loss per observation")


    
    