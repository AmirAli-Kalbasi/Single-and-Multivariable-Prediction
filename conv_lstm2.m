function it = conv_lstm(Train_inputs,Train_targets,Test_inputs,Test_targets,layer_number,bias,filters,batch_size)
% kernel
Wxi = filters.Wxi;
Whi = filters.Whi;
Wci = filters.Wci;
Wxf = filters.Wxf;
Whf = filters.Whf;
Wcf = filters.Wcf;
Wxc = filters.Wxc;
Whc = filters.Whc;
Wxo = filters.Wxo;
Who = filters.Who;
Wco = filters.Wco;
% bias
bi = bias.bi;
bf = bias.bf;
bc = bias.bc;
bo = bias.bo;
% initial values
channel_number = size(Train_inputs,1);
xt = Train_inputs{1,1};
[row_number,column_number] = size(xt);
xt = Train_inputs;
for i=1:layer_number
Ht_prev{i} = zeros(row_number,column_number);
ct_prev{i} = zeros(row_number,column_number);
end
% feed forward
for i=1:layer_number
xt_Wxi = zeros(row_number,column_number);
for channel_count = 1:channel_number
    xt_Wxi = xt_Wxi+conv2(xt{channel_count,1},Wxi{channel_count,i},'same');
end

it{i} = logsig(convn(xt,Wxi{i},'same')+convn(Ht_prev{i},Whi{i},'same')+convn(ct_prev{i},Wci{i},'same')+bi);
ft{i} = logsig(convn(xt,Wxf{i},'same')+convn(Ht_prev{i},Whf{i},'same')+convn(ct_prev{i},Wcf{i},'same')+bf);
ct{i} = ft{i}.* ct_prev{i}+it{i}.*sigmoid_g(convn(xt,Wxc{i},'same')+convn(Ht_prev{i},Whc{i},'same')+bc);
ot{i} = logsig(convn(xt,Wxo{i},'same')+convn(Ht_prev{i},Who{i},'same')+convn(ct{i},Wco{i},'same')+bo);
Ht{i} = ot{i}.*sigmoid_h(ct{i});
Ht_prev{i} = Ht{i};
ct_prev{i} = ct{i};
end
