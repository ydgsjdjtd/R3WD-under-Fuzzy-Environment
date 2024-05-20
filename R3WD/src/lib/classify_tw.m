function model = classify_tw(trainSet, testSet, costMatrix, factor_bp, factor_bn)
    % 将两个 set 转为 matlab 格式
    % trainSetArr = weka2matlab(trainSet,{})
    % testSetArr = weka2matlab(testSet,{})

    numInstance_test = testSet.numInstances();
    numClass_test = testSet.numClasses(); 

    pos_loss = 0;
    bnd_loss = 0;
    neg_loss = 0;

    posMatrix = zeros(numClass_test, numClass_test);
    bndMatrix = zeros(numClass_test, numClass_test);
    negMatrix = zeros(numClass_test, numClass_test);

    % 这里原来是打算用 Cell Array 记录所有 instance 但是不好判断一个 instance 是否存在于 Instances 中，改用记录 index（从1开始的）

    all = 1:numInstance_test;
    posList = [];
    posList_true = [];
    bndList = [];
    negList = [];

    % 训练 J48 Decision Tree
    j48 = wekaTrainModel(trainSet, 'trees.J48');

    proDistribution = zeros(numInstance_test, numClass_test);
    alpha = zeros(numInstance_test, numClass_test);
    beta = zeros(numInstance_test, numClass_test);

    for j = 1:numInstance_test
        proDistribution(j, :) = j48.distributionForInstance(testSet.instance(j - 1));
        alpha(j, :) = threshold_alpha(costMatrix, numClass_test, proDistribution(j, :), factor_bp, factor_bn);
        beta(j, :) = threshold_beta(costMatrix, numClass_test, proDistribution(j, :), factor_bp, factor_bn);
    end

    for j = 1:numInstance_test
        instance = testSet.instance(j - 1);
        trueClass = instance.classValue() + 1;
        for n = 1:numClass_test

            if proDistribution(j, n) >= alpha(j, n) && ~ismember(j, posList) % 判断索引是否存在即可
                posMatrix(n, trueClass) = posMatrix(n, trueClass) + 1;
                posList = [posList, j]; % 添加索引
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
                        loss_bn = sum(factor_bn * costMatrix(n, :)' .* proDistribution(j, :));
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
                    loss_bn = sum(factor_bn * costMatrix(n, :)' .* proDistribution(j, :));
                end
                bnd_loss = bnd_loss + loss_bn + loss_bp;
            end
        end
    end

    all = setdiff(all, [posList, bndList]);
    negList_temp = all;
    
    for k = 1:length(negList_temp)
        index = negList_temp(k);
        instance = testSet.instance(index - 1);

        trueClass = instance.classValue() + 1;
        for n = 1:numClass_test
            if proDistribution(index, n) <= beta(index, n) && ~ismember(index, negList) % ismember 感觉有问题？
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

    model = struct('numInstance_train', trainSet.numInstances(), 'numInstance_test', testSet.numInstances(), ...
                   'numClass', trainSet.numClasses(), 'accuracy', accuracy, 'deferment', deferment, ...
                   'loss', loss, 'posMatrix', posMatrix, 'bndMatrix', bndMatrix, 'negMatrix', negMatrix, ...
                   'posList', posList, 'posList_true', posList_true, 'bndList', bndList, 'negList', negList);

end
