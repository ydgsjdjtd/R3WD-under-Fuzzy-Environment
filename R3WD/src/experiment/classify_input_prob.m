function model = classify_input_prob(proDistribution,labels,costMatrix, factor_bp, factor_bn)
    % ������ set תΪ matlab ��ʽ
    % trainSetArr = weka2matlab(trainSet,{})
    % testSetArr = weka2matlab(testSet,{})
    [numInstance_test,  numClass_test] = size(proDistribution);

    pos_loss = 0;
    bnd_loss = 0;
    neg_loss = 0;

    posMatrix = zeros(numClass_test, numClass_test);
    bndMatrix = zeros(numClass_test, numClass_test);
    negMatrix = zeros(numClass_test, numClass_test);

    % ����ԭ���Ǵ����� Cell Array ��¼���� instance ���ǲ����ж�һ�� instance �Ƿ������ Instances �У����ü�¼ index����1��ʼ�ģ�

    all = 1:numInstance_test;
    posList = [];
    posList_true = [];
    bndList = [];
    negList = [];

    alpha = zeros(numInstance_test, numClass_test);
    beta = zeros(numInstance_test, numClass_test);

    for j = 1:numInstance_test
        alpha(j, :) = threshold_alpha(costMatrix, numClass_test, proDistribution(j, :), factor_bp, factor_bn);
        beta(j, :) = threshold_beta(costMatrix, numClass_test, proDistribution(j, :), factor_bp, factor_bn);
    end

    for j = 1:numInstance_test
        trueClass = labels(j);
        for n = 1:numClass_test

            if proDistribution(j, n) >= alpha(j, n) && ~ismember(j, posList) % �ж������Ƿ���ڼ���
                posMatrix(n, trueClass) = posMatrix(n, trueClass) + 1;
                posList = [posList, j]; % �������
                if n == trueClass
                    posList_true = [posList_true, j];
                end
                if ismember(j, bndList)
                    bndList(bndList == j) = [];
                    loss_bp = 0;
                    loss_bn = 0;
                    if n == trueClass
                        loss_bp = sum(factor_bp * costMatrix(:, trueClass)' .* proDistribution(j, :));
                    else
                        loss_bn = sum(factor_bn * costMatrix(n, :) .* proDistribution(j, :));
                    end
                    bnd_loss = bnd_loss - loss_bn - loss_bp;
                end
            end

            if proDistribution(j, n) > beta(j, n) && proDistribution(j, n) < alpha(j, n) && ~ismember(j, bndList)
                bndMatrix(n, trueClass) = bndMatrix(n, trueClass) + 1;
                bndList = [bndList, j];
                loss_bp = 0;
                loss_bn = 0;
                if n == trueClass
                    loss_bp = sum(factor_bp * costMatrix(:, trueClass)' .* proDistribution(j, :));
                else
                    loss_bn = sum(factor_bn * costMatrix(n, :) .* proDistribution(j, :));
                end
                bnd_loss = bnd_loss + loss_bn + loss_bp;
            end
        end
    end

    all = setdiff(all, [posList, bndList]);
    negList_temp = all;
    
    for k = 1:length(negList_temp)
        index = negList_temp(k);
        trueClass = labels(k);
        for n = 1:numClass_test
            if proDistribution(index, n) <= beta(index, n) && ~ismember(index, negList) % ismember �о������⣿
                negMatrix(n, trueClass) = negMatrix(n, trueClass) + 1;
                negList = [negList, index];
                if n == trueClass
                    sum_loss = sum(costMatrix(:, trueClass)' .* proDistribution(index, :));
                    neg_loss = neg_loss + sum_loss;
                end
            end
        end
    end

    % Calculating the total positive loss
    for k = 1:numClass_test
        for n = 1:numClass_test
            pos_loss = pos_loss + posMatrix(k, n) * costMatrix(k, n);
        end
    end

    % Total loss calculation
    loss = pos_loss + bnd_loss + neg_loss;
    % Calculating accuracy
    true_count = sum(diag(posMatrix)); % Sum of diagonal elements of posMatrix
    accuracy = true_count / length(posList);

    % Calculating deferment
    deferment = (numInstance_test - length(posList)) / numInstance_test;

    model = struct('accuracy', accuracy, 'deferment', deferment, ...
                   'loss', loss, 'posMatrix', posMatrix, 'bndMatrix', bndMatrix, 'negMatrix', negMatrix, ...
                   'posList', posList, 'posList_true', posList_true, 'bndList', bndList, 'negList', negList);
end
