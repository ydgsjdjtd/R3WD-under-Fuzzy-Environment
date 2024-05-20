function result  =  gc_CSCB_MC(test_label, prob, cost_matrix)
% for multi-classfication
% test_label_c:  No. of rows: test_num  列向量
% prob: the classification probabilities, each row is for one sample
%       every column is the probability for one class
% cost_matrix: the misclassification cost matrix, size: n*n
%              n is the number of categories. class_num

% result_2way: the decision, decision cost and error_rate for both methods
%------------------------------------------------------------

% test_label_c = test_label';
% prob = [loop.prob{1,1}]';
%cost_matrix = 2*(ones(M,M)-eye(M,M));
%test------------------------------------

class_num = size(prob,2); % number of categories
    assert(class_num==size(cost_matrix,1),'size of prob and cost_matrix?')
test_num = size(test_label,1);  % number of test samples
    assert(test_num == size(prob,1),'test_label prob ?')

%%CSMC
bayes_loss = prob*cost_matrix;% the expected loss of test samples, bayes loss
[loss_sort, num_sort]=sort(bayes_loss,2,'ascend');
dec_CSMC = num_sort(:,1);
    assert(test_num == size(dec_CSMC,1),'decision CSWC ?')
dec_res_CS = dec_CSMC == test_label;%代价最小的决策
err_CSMC = 1 - sum(dec_res_CS)/test_num;%错误率

cost_i_CS = zeros(size(dec_CSMC));
for i=1:test_num
    cost_i_CS(i) = cost_matrix(test_label(i),dec_CSMC(i));
end
cost_CSMC = sum(cost_i_CS);%总代价

% sum_rate = 0;
% for i=1:class_num
%     class_r = sum(dec_CS3WD((eachclass_num*(i-1)+1):(eachclass_num*i),1).*pos_reg((eachclass_num*(i-1)+1):(eachclass_num*i)) ...
%         == test_label_c((eachclass_num*(i-1)+1):(eachclass_num*i)));
%     class_pos = sum(dec_CS2WD((eachclass_num*(i-1)+1):(eachclass_num*i)));
%     class_rate = class_r / class_pos;
%     sum_rate = sum_rate + class_rate;
% end
% ave_rate = sum_rate / class_num;%CCA / CAR

%% CBMC
[prob_sort,cate_sort]=sort(prob,2,'descend'); 
dec_CBMC = cate_sort(:,1);
    assert(test_num == size(dec_CBMC,1),'decision CBMC ?')
dec_res_CBMC = dec_CBMC == test_label;%概率最大的决策
err_CBMC = 1- sum(dec_res_CBMC)/test_num;%错误率
% right_P2_num = sum( cate_sort(:,2)==test_label_c)
% the second most possible choice

cost_i_CBMC = zeros(size(test_label));
for i=1:test_num
    cost_i_CBMC(i) = cost_matrix(test_label(i),dec_CBMC(i));
end
cost_CBMC = sum(cost_i_CBMC);%总代价

%% output
result.dec = [dec_CSMC,dec_CBMC];
result.cost = [cost_CSMC cost_CBMC];
result.err = [err_CSMC err_CBMC];
result.bayes_risk = bayes_loss;
result.ord = 'CSMC CBMC';