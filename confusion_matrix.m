function [] = confusion_matrix(labels,predictions)

    M = size(unique(labels),1);
    N = size(labels,1);
    targets = zeros(M,N);
    outputs = zeros(M,N);
    for i=1:N
       targets(labels(i)+1, i) = 1; 
       outputs(predictions(i)+1, i) = 1; 
    end

    plotconfusion(targets,outputs)
end
