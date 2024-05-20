% 创建一个随机矩阵
function randArr = randArray(n, randNum)
    rng(randNum); 
    randArr = 10 * rand(n, n); 
    for i = 1:n
        randArr(i, i) = 0;
    end
end
