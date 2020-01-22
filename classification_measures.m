function [precision, recall, F1] = classification_measures(labels, predictions)
    % Function that return precision, recall and F1-score for multiclass
    % classification problem


    
    CM = confusionmat(labels,predictions);
    
    recall = mean(diag(CM)./sum(CM,2));
    precision = mean(diag(CM)./sum(CM)');
    
    F1 = 2*(recall*precision)/(recall+precision);
    
end

