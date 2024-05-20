function confusionMatrix = lt_confuse_matrix(trueLabels, predictedLabels)

    % 确保输入参数的长度一致

    assert(length(trueLabels) == length(predictedLabels), '输入参数长度不一致');



    % 获取所有类别

    uniqueClasses = unique(trueLabels);



    % 初始化混淆矩阵

    numClasses = length(uniqueClasses);

    confusionMatrix = zeros(numClasses);



    % 遍历每个样本，更新混淆矩阵

    for i = 1:length(trueLabels)

        trueClass = find(uniqueClasses == trueLabels(i));

        predictedClass = find(uniqueClasses == predictedLabels(i));

        confusionMatrix(trueClass, predictedClass) = confusionMatrix(trueClass, predictedClass) + 1;

    end



end