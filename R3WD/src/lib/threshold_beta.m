% 计算阈值:beta
function beta = threshold_beta(matrix, n, pro, factor_bp, factor_bn)
    beta = zeros(1, n);
    sum_over = zeros(1, n);
    sum_down = zeros(1, n);
    for i = 1:n
        sum_over(i) = 0;
        sum_down(i) = 0;
        for j = 1:n
            if i ~= j
                sum_over(i) = sum_over(i) + factor_bn * pro(j) * matrix(i, j);
                sum_down(i) = sum_down(i) + pro(j) * matrix(j, i) * (1 - factor_bp);
            end
        end
        sum_down(i) = sum_down(i) + sum_over(i);
        if sum_down(i) ~= 0
            beta(i) = sum_over(i) / sum_down(i);
        else
            beta(i) = 0;
        end
    end
end
