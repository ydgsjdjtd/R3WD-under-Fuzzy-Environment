function [E] = lt_add_label_noise(data,ratio)
    option = ['-P ',num2str(ratio)];
    addNoise = wekaFilter('unsupervised.attribute.AddNoise', option);
    [E, ~] = wekaApplyFilter(data, addNoise);
end
