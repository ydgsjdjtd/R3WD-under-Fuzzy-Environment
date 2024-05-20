fileroot='../../../newdata/';
filelist = dir(fullfile(fileroot,'*.arff'));
num_ds = length(filelist);
rng(1);
datasets = cell(num_ds+1,1);
cms = cell(num_ds+1,1);
datasets(num_ds+1) = {'<cls>'};
cms(num_ds+1) = {'<cls>'};
for j=1:1:num_ds
    fname = filelist(j).name;
    %% N-fold cross-validation
    D = wekaLoadData(strcat(fileroot,fname));
    fname = split(fname,".");
 
    num_classes = D.numClasses;
    cm = rand(num_classes,num_classes);
    cm = cm.*2;
    for i=1:1:num_classes
        cm(i,i) = 0;
    end
    
    cms(j) = {mat2cell(cm,num_classes)};
    datasets(j) = fname(1);
end
x = struct2table(struct('dataset', datasets, 'cm', cms));
writelines(jsonencode(x),"./json/"+"jsondata"+".json");