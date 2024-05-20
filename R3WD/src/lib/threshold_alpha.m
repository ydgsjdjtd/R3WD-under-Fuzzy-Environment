% 计算阈值:alpha
function alpha = threshold_alpha(matrix, n, pro, factor_bp, factor_bn)
    alpha = zeros(1, n);
    sum_over = zeros(1, n);
    sum_down = zeros(1, n);
    for i = 1:n
        sum_over(i) = 0;
        sum_down(i) = 0;
        for j = 1:n
            if i ~= j
                sum_over(i) = sum_over(i) + pro(j) * matrix(i, j) * (1 - factor_bn);
                sum_down(i) = sum_down(i) + pro(j) * matrix(j, i) * factor_bp;
            end
        end
        sum_down(i) = sum_down(i) + sum_over(i);
        if sum_down(i) ~= 0
            alpha(i) = sum_over(i) / sum_down(i);
        else
            alpha(i) = 0.5;
        end
    end
end