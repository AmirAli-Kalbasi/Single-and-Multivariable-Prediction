close all;clear all;clc;
%% Import data
Data_all = importdata('Data.xlsx');
%Data = Data_all.data(1:3000);
Data = Data_all(1:1000);
Data = Data(~isnan(Data));
clear Data_all
%% Sliding window
window_size = 12;
% Input
Inputs = zeros(size(Data,1)-window_size,window_size);
for i=1:size(Data,1)-window_size
    Inputs(i,:)=Data(i:i+window_size-1);
end
% Target
Targets = [Inputs(2:end,1);Data(end)];
%% CWT
% fb = cwtfilterbank('SignalLength',numel(Data),'SamplingFrequency',1000,'FrequencyLimits',[0.5 100]);
FB =  cwtfilterbank('SignalLength',numel(Inputs(1,:)));
size_cwt = size(cwt(Inputs(1,:),'FilterBank',FB),1);
CWT_data.cfs = cell(1,size(Data,1)-window_size);
[~,CWT_data.f] = cwt(Inputs(1,:),'FilterBank',FB);
for i=1:size(Data,1)-window_size
    CWT_data.cfs{i} =abs(cwt(Inputs(i,:),'FilterBank',FB));
end
% Plot
% surface((1:window_size),CWT_data.f,abs(CWT_data.cfs{1}))
%% Test, Train
Train_number = round(length(CWT_data.cfs)*0.7);
Train_inputs = CWT_data.cfs(1:Train_number);
Train_targets = Targets(1:Train_number);
Test_inputs = CWT_data.cfs(1+Train_number:length(CWT_data.cfs));
Test_targets = Targets(1+Train_number:length(CWT_data.cfs));
%% 
kernel_number = 200;
kernel_width = 3;
kernel_length = 3;
channel_number = size(Train_inputs,1);
filters = initial_kernel(kernel_number,kernel_width,kernel_length,channel_number);
layer_number = 200;
[input_size.w,input_size.l] = size(Train_inputs{1,1});
bias = initial_bias(input_size,layer_number);
batch_size = 5;
epoch = 2;
[Ht1,Ht_test1] = conv_lstm(Train_inputs,Train_targets,Test_inputs,Test_targets,layer_number,bias,filters,batch_size,epoch);
%%
[m1,n1] = size(Ht1);
for m =1:m1
for n=1:n1
H1_pool{m,n} = max_pool(Ht1{m,n},2,2);
end
end
[m1,n1] = size(Ht_test1);
for m =1:m1
for n=1:n1
H1_test_pool{m,n} = max_pool(Ht_test1{m,n},2,2);
end
end
%%
kernel_number = 200;
kernel_width = 3;
kernel_length = 3;
channel_number = size(H1_pool,1);
filters = initial_kernel(kernel_number,kernel_width,kernel_length,channel_number);
layer_number = 200;
[input_size.w,input_size.l] = size(H1_pool{1,1});
bias = initial_bias(input_size,layer_number);
batch_size = 5;
epoch = 2;
[Ht2,Ht_test2] = conv_lstm(H1_pool,Train_targets,H1_test_pool,Test_targets,50,bias,filters,batch_size,epoch);
%%
[m2,n2] = size(Ht2);
for m =1:m2
for n=1:n2
H2_pool{m,n} = max_pool(Ht2{m,n},2,2);
end
end
[m1,n1] = size(Ht_test2);
for m =1:m1
for n=1:n1
H2_test_pool{m,n} = max_pool(Ht_test2{m,n},2,2);
end
end
%%
kernel_number = 50;
kernel_width = 3;
kernel_length = 3;
channel_number = size(H2_pool,1);
filters = initial_kernel(kernel_number,kernel_width,kernel_length,channel_number);
layer_number = 50;
[input_size.w,input_size.l] = size(H2_pool{1,1});
bias = initial_bias(input_size,layer_number);
batch_size = 5;
epoch = 2;
[Ht3,Ht_test3] = conv_lstm(H2_pool,Train_targets,H2_test_pool,Test_targets,1,bias,filters,batch_size,epoch);

[m3,n3] = size(Ht3);
for m =1:m3
for n=1:n3
H3_pool{m,n} = max_pool(Ht3{m,n},2,2);
end
end
[m3,n3] = size(Ht_test3);
for m =1:m3
for n=1:n3
H3_test_pool{m,n} = max_pool(Ht_test3{m,n},2,2);
end
end
%%
NN_parameters.neurons_number = [10 1];
NN_parameters.eta = [0.2 0.1];
NN_parameters.epochs = 2;
layers_number = length(NN_parameters.neurons_number);
NN_parameters.Activation_Function = {'logsig','purelin'};
[y_train,y_test,output_train,MSE_train,output_test,MSE_test] = predict_data(Ht3,Train_targets,Ht_test3,Test_targets,NN_parameters);










