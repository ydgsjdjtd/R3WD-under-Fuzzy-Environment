% fileroot='../../data10/';
fileroot='../../../newdata/';


filelist = dir(fullfile(fileroot,'*.arff'));
num_ds = length(filelist);
percentage = 10;
% Number of folds
N = 10;
for j=1:1:num_ds
    df = cell(N+1,1);
    fname = filelist(j).name;

    %% N-fold cross-validation
    D = wekaLoadData(strcat(fileroot,fname));

    % Pre-allocate error array
    errors = zeros(1,N); 
    
    % Stratifies a set of instances according to its class values if the class 
    % attribute is nominal (so that afterwards a stratified cross-validation can be performed).
    D.stratify(N);
    
    % Alternatively, randomize the data
    % D.randomize(java.util.Random(1998));
    
    D = lt_add_label_noise(D,percentage);

    for i = 1:N
        test = D.testCV(N, i-1);
        train = D.trainCV(N, i-1);
        % Train model 
        model = wekaTrainModel(train, 'trees.J48');
        % Classify test data 
        [predicted, classProbs, confusionMatrix] = wekaClassify(test,model);
        test_label = test.attributeToDoubleArray(test.classIndex);
        data = [classProbs test_label];
        
        df{i} = data;
    end
    df{N+1} = '<CLS>';
    fullname = split(fname,'.');
    writelines(jsonencode(df),"./label_noise/"+string(percentage)+"/"+fullname(1)+".json")
end
