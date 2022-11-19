function [y_train,y_test,output_train,MSE_train,output_test,MSE_test] = predict_data(Inputs_train,targets_train,Inputs_test,targets_test,NN_parameters)
aa = length(Inputs_train);
for i =1:aa
inputs_train = Inputs_train{i};
[m,n] = size(inputs_train);
if n>1
for N = 1:n
Inputs((N-1)*m+1:(N)*m,i) = inputs_train(1:m,N);
end
end 
end
X_train = Inputs';
Y_train = targets_train;
aa = length(Inputs_test);
for i =1:aa
inputs_test = Inputs_train{i};
[m,n] = size(inputs_test);
if n>1
for N = 1:n
Inputs_test2((N-1)*m+1:(N)*m,i) = inputs_test(1:m,N);
end
end 
end
X_test = Inputs_test2';
Y_test = targets_test;
XX = [X_train;X_test];
YY = [Y_train;Y_test];
method_normalization.name = 'range';
method_normalization.lower_bound = -1;
method_normalization.upper_bound = 1;
Data = Normalized([XX,YY],method_normalization);
X = Data(:,1:end-1);
Y = Data(:,end);
x_train = X(1:length(X_train),:);
y_train = Y(1:length(Y_train),:);
x_test = X(1+length(X_train):end,:);
y_test = Y(1+length(Y_train):end,:);
%% Number of layers, neurons
Activation_Function = NN_parameters.Activation_Function;
layers_number = length(NN_parameters.neurons_number);
neurons_number(1) = size(x_train,2);
for i_layers_number = 1:layers_number
    neurons_number(i_layers_number+1) = NN_parameters.neurons_number(i_layers_number);
end
eta = NN_parameters.eta;
epochs = NN_parameters.epochs;   
%% Weights
for i_layers_number = 1:layers_number
    W{i_layers_number} = unifrnd(-1,1,[neurons_number(i_layers_number),neurons_number(i_layers_number+1)]);
end
%% 
MSE_train = [];
MSE_test = [];
for i_epoch = 1:epochs
    sqr_err_epoch_train = [];
    sqr_err_epoch_test = [];
    output_train = [];
    output_test = [];
    for j_dimension = 1:size(x_train,1)
        % Feed-Forward
        net{1} =  x_train(j_dimension,:)*W{1};
        if strcmp(Activation_Function{1},'logsig')==1
        O{1} = logsig(net{1});
        f_driviate{1} = diag(O{1}.*(1-O{1}));
        elseif strcmp(Activation_Function{1},'tanh')==1
        O{1} = tanh(net{1});
        f_driviate{1} = diag(1-O{1}.^2);
        elseif strcmp(Activation_Function{1},'purelin')==1
        O{1} = purelin(net{1});
        f_driviate{1} = diag(ones(1,size(W{1},2)));
        end
        for i_layers_number = 2:layers_number
            net{i_layers_number} =  O{i_layers_number-1}*W{i_layers_number};
            if strcmp(Activation_Function{i_layers_number},'logsig')==1
            O{i_layers_number} = logsig(net{i_layers_number});
            f_driviate{i_layers_number} = diag(O{i_layers_number}.*(1-O{i_layers_number}));
            elseif strcmp(Activation_Function{i_layers_number},'tanh')==1
            O{i_layers_number} = tanh(net{i_layers_number});
            f_driviate{i_layers_number} = diag(1-O{i_layers_number}.^2);
            elseif strcmp(Activation_Function{i_layers_number},'purelin')==1
            O{i_layers_number} = purelin(net{i_layers_number});
            f_driviate{i_layers_number} = diag(ones(1,size(W{i_layers_number},2)));
            end
        end
        output_train(j_dimension,:) = O{end};
        err = y_train(j_dimension,:)-O{end};
        sqr_err_epoch_train(j_dimension) = err^2;
        inputs_layer{1} = eta(1)*(-1)*x_train(j_dimension,:)'*err*f_driviate{end};
        for i_layers_number = 2:layers_number
            inputs_layer{i_layers_number} = eta(i_layers_number)*(-1)*O{i_layers_number-1}'*err*f_driviate{end};
        end
        for i_layers_number = layers_number:-1:2
            weights{i_layers_number} = W{i_layers_number}'*f_driviate{i_layers_number-1};
        end
        weights{1} = 1;
        % Back propagation
        for i_layers_number = 1:layers_number
            clear save_multiply save_multiply1 
            save_multiply = inputs_layer{i_layers_number};
            save_multiply1 = 1;
            for j_layers_number = layers_number:-1:i_layers_number
                if i_layers_number==j_layers_number
                save_multiply1 = save_multiply1;
                else
                save_multiply1 = save_multiply1*weights{j_layers_number};
                end
            end
            save_multiply = save_multiply*save_multiply1;
            W{i_layers_number} = W{i_layers_number}-save_multiply;
        end
    end
    mse_epoch_train = 0.5*sum(sqr_err_epoch_train)/size(x_train,1);
    MSE_train(i_epoch) = mse_epoch_train;
    for k_test = 1:size(x_test,1)
        % Feed-Forward
        net{1} = x_test(k_test,:)*W{1};
        if strcmp(Activation_Function{1},'logsig')==1
        O{1} = logsig(net{1});
        f_driviate{1} = diag(O{1}.*(1-O{1}));
        elseif strcmp(Activation_Function{1},'tanh')==1
        O{1} = tanh(net{1});
        f_driviate{1} = diag(1-O{1}.^2);
        elseif strcmp(Activation_Function{1},'purelin')==1
        O{1} = purelin(net{1});
        f_driviate{1} = diag(ones(1,size(W{1},2)));
        end
        for i_layers_number = 2:layers_number
            net{i_layers_number} =  O{i_layers_number-1}*W{i_layers_number};
            if strcmp(Activation_Function{i_layers_number},'logsig')==1
            O{i_layers_number} = logsig(net{i_layers_number});
            f_driviate{i_layers_number} = diag(O{i_layers_number}.*(1-O{i_layers_number}));
            elseif strcmp(Activation_Function{i_layers_number},'tanh')==1
            O{i_layers_number} = tanh(net{i_layers_number});
            f_driviate{i_layers_number} = diag(1-O{i_layers_number}.^2);
            elseif strcmp(Activation_Function{i_layers_number},'purelin')==1
            O{i_layers_number} = purelin(net{i_layers_number});
            f_driviate{i_layers_number} = diag(ones(1,size(W{i_layers_number},2)));
            end
        end
        output_test(k_test,:) = O{end};  
        % Error
        err = y_test(k_test,:)-O{end};
        sqr_err_epoch_test(k_test) = err^2;
    end
    mse_epoch_test = 0.5*sum(sqr_err_epoch_test)/size(x_test,1);
    MSE_test(i_epoch) = mse_epoch_test;
end
end
