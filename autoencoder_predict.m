close all;clear all;clc;
%% Import data
Data_all = importdata('Data.csv');
Data = Data_all.data(1:3000);
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
Train_targets = CWT_data.cfs(2:1+Train_number);
Test_inputs = CWT_data.cfs(1+Train_number:length(CWT_data.cfs)-1);
Test_targets = CWT_data.cfs(2+Train_number:length(CWT_data.cfs));
% %% 
kernel_number = 200;
kernel_width = 3;
kernel_length = 3;
channel_number = size(Train_inputs,1);
filters = initial_kernel(kernel_number,kernel_width,kernel_length,channel_number);
layer_number = 200;
[input_size.w,input_size.l] = size(Train_inputs{1,1});
bias = initial_bias(input_size,layer_number);
batch_size = 5;
epoch = 20;
[out_decoder,out_decoder_test] = conv_lstm_predict(Train_inputs,Train_targets,Test_inputs,Test_targets,layer_number,bias,filters,batch_size,epoch);













