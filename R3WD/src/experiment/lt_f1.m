function [f1] = lt_f1(test_label,pred_label)
    M=lt_confuse_matrix(test_label,pred_label);
    
    precision = diag(M)./(sum(M,2) + 0.0001);  %按列求和: TP/(TP+FP)
    
    recall = diag(M)./(sum(M,1)+0.0001)'; %按行求和: TP/(TP+FN)
    
    precision = mean(precision);
    
    recall = mean(recall);
    
    f1 = 2*precision*recall/(precision + recall);
end

