function result  =  gc_cs3wmc(prob, cost_matrix, a_trust_i,test_label)
% input-----------------------------------------
% prob: every row is the probability vector of one test sample
% No. of rows: No. of test samples  No. of Columns: No. of categories
% cost_matrix: \lamda_ij, real i, predict j
%  a_trust: the given trust threshold
% test_label_c:  No. of rows: test_num  列向量, start from 1

% output-----------------------------------------------
% bayes_loss, every row is for a test sample, # columns = categories
% dec_CS3WD, decision for samples: positive decision, the second column 0
%                       boundary decision, the second column >0
% mis_cost, meaning of every column: positive decision cost, boundary decision cost
%                      boundary decision,  cost_OPD, cost_SOPD 

%test-----------------------------------------------
% test_label_c = test_label';
% prob = [loop.prob{2,8}]';
% cost_matrix = 10*rand(M,M).*(ones(M,M) - eye(M,M));

%% bayes and theta
class_num = size(prob,2); % number of categories
    assert(class_num==size(cost_matrix,1), 'size of prob and cost_matrix ?')
test_num = size(test_label,1);  % number of test samples
    assert(test_num == size(prob,1), 'test_label prob ?')
bayes_loss = prob*cost_matrix; % the expected loss of test samples
[loss_sort,num_sort]=sort(bayes_loss, 2, 'ascend'); % sort the bayes loss in every row
theta_samps = (loss_sort(:,2)-loss_sort(:,1))./loss_sort(:,2);% theta: trust level of samples

%% a_trust and 3WD
a_trust = prctile(theta_samps,a_trust_i);% 25%,lower quartile;
% disp("==")
% disp(a_trust)
% disp("==")
    assert(a_trust>0, 'alpha > 0 ?')
pos_reg = theta_samps > a_trust;% samples in positive region, logical, 1 positive, 0 boundary
num_P = size(find(pos_reg),1);%判断为Pos域的样本数
boun_reg = 1 - pos_reg;
% disp(boun_reg)
num_B = size(find(boun_reg),1);%判断为Bnd域的样本数
    assert(num_P+num_B == test_num, 'NUMBER_P ?')
dec_cs3wmc = [num_sort(:,1), num_sort(:,2).*boun_reg];
% disp(num_sort)
% disp(size(pos_reg))
% disp(size(dec_cs3wmc))
% dec_CS3WD = int8([num_sort(:,1), num_sort(:,2).*boun_reg]);
% decision for samples: positive decision, the second column 0
%                       boundary decision, the second column >0
% assert(sum(pred2==dec_CS3WD(:,1)')==size(dec_CS2WD,1),'cost-insensitive right?')

right_P = sum(dec_cs3wmc(:,1).*pos_reg == test_label);%确定性判断正确的样本数
err_P = 1 - right_P/num_P;%确定性判断错误率 the error rate in positive decisions

right_B1 = sum(dec_cs3wmc(:,1).*boun_reg == test_label);%边界域首选项正确的样本数
right_B2 = sum(dec_cs3wmc(:,2) == test_label);%边界域次选项正确的样本数
right_B = right_B1 + right_B2;%边界域正确的样本数

dec_vec = zeros(size(dec_cs3wmc(:,1)));
dec_vec(:) = dec_cs3wmc(:,1).*pos_reg;
for i=1:1:length(dec_vec)
    B1 = (dec_cs3wmc(:,1).*boun_reg == test_label);
    if B1(i)
        dec_vec(i) = test_label(i);
    end
    B2 = (dec_cs3wmc(:,2) == test_label);
    if B2(i)
        dec_vec(i) = test_label(i);
    end
end

wrong_B1= num_B-right_B1;%边界域首选项错误的样本数
wrong_B2 = num_B-right_B2;%边界域次选项错误的样本数

err_B1 = wrong_B1 / num_B;%边界域首选项错误率
err_B2 = wrong_B2 / num_B;%边界域次选项错误率
% for bondary decision, it is right if  OPD or SOPD is right
err_B = 1 - right_B/num_B;%边界域错误率
err_3WD = 1 - (right_P + right_B)/test_num;%综合错误率

cost_2opt = zeros(size(dec_cs3wmc));
% disp(size(dec_cs3wmc))
for j=1:test_num
    cost_2opt(j,:) = cost_matrix(test_label(j),num_sort(j,1:2));% the misclassification cost for OPD and SOPD
    %assert(cost_2opt(i,2) == cost_matrix(test_label_c(i),num_sort(i,2)), 'c')
end
cost_P = cost_2opt(:,1).*pos_reg;% if boundary decision, value is 0
cost_B = (cost_2opt(:,1)+cost_2opt(:,2)).*boun_reg/(2 + a_trust);% the boundary decision cost, if positive decision, value is 0
% cost_B =1.*boun_reg;
%cost_B = (cost_2opt(:,1)+cost_2opt(:,2)).*boun_reg/4;

cost_P_B = cost_P + cost_B;
cost_3wd = sum(cost_P_B);%总代价
%     assert(sum(cost_boun) + sum(cost_pos) < sum(cost_2opt(:,1)),'3WD useless')


%% output
result.theta_samps = theta_samps;
result.pos_reg = pos_reg;%正域
result.dec = dec_cs3wmc;%决策结果
result.cost_vec = [cost_2opt cost_P_B];%代价
% disp(size(cost_2opt))
% disp(size(cost_P_B))
result.cost_vec_arg = 'cost_2opt(2c) cost_P_B';
result.cost_err = [cost_3wd err_P err_B err_3WD err_B1 err_B2];
%1总代价 2确定性判断错误率 3边界域错误率 4综合错误率 5边界域首选项错误率 6边界域次选项错误率
result.cost_err_arg = 'cost_3wd, err_P, err_B, err_3WD...';
result.para = [a_trust num_P num_B];
result.ord = 'a_trust num_P num_B';
result.dec_vec = dec_vec;
% dec_reg = 2 - pos_reg;