function [Ht,Ht_test] = conv_lstm(Train_inputs,Train_targets,Test_inputs,Test_targets,layer_number,bias,filters,batch_size,epoch)
eta1 = 0.000000001;
eta2 = 0.000000001;
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
input_number = size(Train_inputs,2);
[row_number,column_number] = size(Train_inputs{1,1});
xt = Train_inputs;

% feed forward
for epoch_count = 1:epoch
for i=1:layer_number
Ht_prev{i} = zeros(row_number,column_number);
ct_prev{i} = zeros(row_number,column_number);
end
for input_count = 1:input_number
e =  0;
% Encoder
for i=1:layer_number
xt_Wxi = zeros(row_number,column_number);
xt_Wxf = zeros(row_number,column_number);
xt_Wxc = zeros(row_number,column_number);
xt_Wxo = zeros(row_number,column_number);
for channel_count = 1:channel_number
    xt_Wxi = xt_Wxi+conv2(xt{channel_count,input_count},Wxi{channel_count,i},'same');
    xt_Wxf = xt_Wxf+conv2(xt{channel_count,input_count},Wxf{channel_count,i},'same');
    xt_Wxc = xt_Wxc+conv2(xt{channel_count,input_count},Wxc{channel_count,i},'same');
    xt_Wxo = xt_Wxo+conv2(xt{channel_count,input_count},Wxo{channel_count,i},'same');
end
it{i,input_count} = logsig(xt_Wxi+conv2(Ht_prev{i},Whi{i},'same')+conv2(ct_prev{i},Wci{i},'same')+bi{i});
ft{i,input_count} = logsig(xt_Wxf+conv2(Ht_prev{i},Whf{i},'same')+conv2(ct_prev{i},Wcf{i},'same')+bf{i});
chatt{i,input_count} = sigmoid_g(xt_Wxc+conv2(Ht_prev{i},Whc{i},'same')+bc{i});
ct{i,input_count} = ft{i,input_count}.* ct_prev{i}+it{i,input_count}.*chatt{i,input_count};
ot{i,input_count} = logsig(xt_Wxo+conv2(Ht_prev{i},Who{i},'same')+conv2(ct_prev{i},Wco{i},'same')+bo{i});
Ht{i,input_count} = ot{i,input_count}.*sigmoid_h(ct{i,input_count});
Ht_prev{i} = Ht{i,input_count};
ct_prev{i} = ct{i,input_count};
end
% Decoder
for decoder_kernel_number = 1:channel_number
out_decoder_temp = zeros(row_number,column_number);
for decoder_channel_number=1:layer_number
out_decoder_temp = out_decoder_temp + conv2(Ht{decoder_channel_number,input_count},filters.Wdecoder{decoder_channel_number,decoder_kernel_number},'same');
end
out_decoder{decoder_kernel_number,input_count} = out_decoder_temp+bias.decoder{decoder_kernel_number};
end
for channel_number_count = 1:channel_number
e = e+immse(xt{channel_number_count,input_count},out_decoder{channel_number_count,input_count});
end
E(input_count) = 0.5*e/channel_number_count;
% Update decoder
for decoder_channel_number=1:layer_number
    for decoder_kernel_number = 1:channel_number
        HH = zeros(2+size(Ht{decoder_channel_number,input_count}));
        HH(2:end-1,2:end-1) = Ht{decoder_channel_number,input_count};
        filters.Wdecoder{decoder_channel_number,decoder_kernel_number} = filters.Wdecoder{decoder_channel_number,decoder_kernel_number} + ...
            eta1*conv2(HH,(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}),'valid'); 
    end
end
for decoder_kernel_number = 1:channel_number
   bias.decoder{decoder_kernel_number} = bias.decoder{decoder_kernel_number}+eta1*(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number});
end
% update decoder

if input_count>2
for decoder_channel_number=1:layer_number
for decoder_kernel_number = 1:channel_number
HH = zeros(2+size(Ht{decoder_channel_number,input_count-1}));
HH(2:end-1,2:end-1) = Ht{decoder_channel_number,input_count-1};
XT  = zeros(2+size(xt{decoder_kernel_number,input_count}));
XT(2:end-1,2:end-1) = xt{decoder_kernel_number,input_count};
CT  = zeros(2+size(ct{layer_number,input_count-1}));
CT(2:end-1,2:end-1) = ct{layer_number,input_count-1};

HH_prev = zeros(2+size(Ht{decoder_channel_number,input_count-2}));
HH_prev(2:end-1,2:end-1) = Ht{decoder_channel_number,input_count-2};
XT_prev  = zeros(2+size(xt{decoder_kernel_number,input_count-1}));
XT_prev(2:end-1,2:end-1) = xt{decoder_kernel_number,input_count-1};
CT_prev  = zeros(2+size(ct{layer_number,input_count-2}));
CT_prev(2:end-1,2:end-1) = ct{layer_number,input_count-2};

% i
Wxi{decoder_kernel_number,decoder_channel_number} = Wxi{decoder_kernel_number,decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count})...
  .*(ot{decoder_channel_number,input_count}.*(1-(tanh(ct{decoder_channel_number,input_count})).^2)),'same').*...
  (conv2(XT_prev,ft{decoder_channel_number,input_count}.*chatt{decoder_channel_number,input_count-1}.*(it{decoder_channel_number,input_count-1}.*(1-it{decoder_channel_number,input_count-1})),'valid')+...
  conv2(XT,chatt{decoder_channel_number,input_count}.*(it{decoder_channel_number,input_count}.*(1-it{decoder_channel_number,input_count})),'valid'));
Whi{decoder_channel_number} = Whi{decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count})...
  .*(ot{decoder_channel_number,input_count}.*(1-(tanh(ct{decoder_channel_number,input_count})).^2)),'same').*...
  (conv2(HH_prev,ft{decoder_channel_number,input_count}.*chatt{decoder_channel_number,input_count-1}.*(it{decoder_channel_number,input_count-1}.*(1-it{decoder_channel_number,input_count-1})),'valid')+...
  conv2(HH,chatt{decoder_channel_number,input_count}.*(it{decoder_channel_number,input_count}.*(1-it{decoder_channel_number,input_count})),'valid'));
Wci{decoder_channel_number} = Wci{decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count})...
  .*(ot{decoder_channel_number,input_count}.*(1-(tanh(ct{decoder_channel_number,input_count})).^2)),'same').*...
  (conv2(CT_prev,ft{decoder_channel_number,input_count}.*chatt{decoder_channel_number,input_count-1}.*(it{decoder_channel_number,input_count-1}.*(1-it{decoder_channel_number,input_count-1})),'valid').*...
  conv2(CT,chatt{decoder_channel_number,input_count}.*(it{decoder_channel_number,input_count}.*(1-it{decoder_channel_number,input_count})),'valid'));
bi{decoder_channel_number} = bi{decoder_channel_number} + ...
  eta2*conv2((xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count})...
  .*(ot{decoder_channel_number,input_count}.*(1-(tanh(ct{decoder_channel_number,input_count})).^2)),filters.Wdecoder{decoder_channel_number,decoder_kernel_number},'same').*...
  (ft{decoder_channel_number,input_count}.*chatt{decoder_channel_number,input_count-1}.*(it{decoder_channel_number,input_count-1}.*(1-it{decoder_channel_number,input_count-1}))).*...
  chatt{decoder_channel_number,input_count}.*(it{decoder_channel_number,input_count}.*(1-it{decoder_channel_number,input_count}));
% o
Wxo{decoder_kernel_number,decoder_channel_number} = Wxo{decoder_kernel_number,decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*...
  (tanh(ct{decoder_channel_number,input_count})),'same').*(conv2(XT,ot{decoder_channel_number,input_count}.*(1-(ot{decoder_channel_number,input_count})),'valid'));
Who{decoder_channel_number} = Who{decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*(tanh(ct{decoder_channel_number,input_count})),'same').*...
  (conv2(HH,ot{decoder_channel_number,input_count}.*(1-(ot{decoder_channel_number,input_count})),'valid'));
Wco{decoder_channel_number} = Wco{decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},((xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*tanh(ct{decoder_channel_number,input_count})),'same').*...
  (conv2(CT,ot{decoder_channel_number,input_count}.*(1-(ot{decoder_channel_number,input_count})),'valid'));
bo{decoder_channel_number} = bo{decoder_channel_number} + ...
  eta2*conv2(((xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*tanh(ct{decoder_channel_number,input_count})),filters.Wdecoder{decoder_channel_number,decoder_kernel_number},'same').*...
  (ot{decoder_channel_number,input_count}.*(1-(ot{decoder_channel_number,input_count})));
% f
Wxf{decoder_kernel_number,decoder_channel_number} = Wxf{decoder_kernel_number,decoder_channel_number} + ...
  (conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*(ot{decoder_channel_number,input_count}.*(1-tanh(ct{decoder_channel_number,input_count}))),'same')).*...
  conv2(XT,ct{decoder_channel_number,input_count-1}.*(ft{decoder_channel_number,input_count}.*(1-ft{decoder_channel_number,input_count})),'valid');
Whf{decoder_channel_number} = Whf{decoder_channel_number} + ...
  (conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*(ot{decoder_channel_number,input_count}.*(1-tanh(ct{decoder_channel_number,input_count}))),'same')).*...
  conv2(HH,ct{decoder_channel_number,input_count-1}.*(ft{decoder_channel_number,input_count}.*(1-ft{decoder_channel_number,input_count})),'valid');
Wcf{decoder_channel_number} = Wcf{decoder_channel_number} + ...
  (conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*(ot{decoder_channel_number,input_count}.*(1-tanh(ct{decoder_channel_number,input_count}))),'same')).*...
  conv2(CT,ct{decoder_channel_number,input_count-1}.*(ft{decoder_channel_number,input_count}.*(1-ft{decoder_channel_number,input_count})),'valid');
bf{decoder_channel_number} = bf{decoder_channel_number} + ...
  (conv2((xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*(ot{decoder_channel_number,input_count}.*(1-tanh(ct{decoder_channel_number,input_count}))),filters.Wdecoder{decoder_channel_number,decoder_kernel_number},'same')).*...
  ct{decoder_channel_number,input_count-1}.*(ft{decoder_channel_number,input_count}.*(1-ft{decoder_channel_number,input_count}));
% c
Wxc{decoder_kernel_number,decoder_channel_number} = Wxc{decoder_kernel_number,decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*...
  (ot{decoder_channel_number,input_count}.*(1-(tanh(ct{decoder_channel_number,input_count})).^2)),'same').* ...
  (conv2(XT_prev,ft{decoder_channel_number,input_count-1}.*it{decoder_channel_number,input_count-1}.*(1-ct{decoder_channel_number,input_count-1}.^2),'valid')+...
  conv2(XT,it{decoder_channel_number,input_count}.*(1-ct{decoder_channel_number,input_count}.^2),'valid'));
Whc{decoder_channel_number} = Whc{decoder_channel_number} + ...
  eta2*conv2(filters.Wdecoder{decoder_channel_number,decoder_kernel_number},(xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*...
  (ot{decoder_channel_number,input_count}.*(1-(tanh(ct{decoder_channel_number,input_count})).^2)),'same').* ...
  (conv2(HH_prev,ft{decoder_channel_number,input_count-1}.*it{decoder_channel_number,input_count-1}.*(1-ct{decoder_channel_number,input_count-1}.^2),'valid')+...
  conv2(HH,it{decoder_channel_number,input_count}.*(1-ct{decoder_channel_number,input_count}.^2),'valid'));
bc{decoder_channel_number} = bc{decoder_channel_number} + ...
  eta2*conv2((xt{decoder_kernel_number,input_count}-out_decoder{decoder_kernel_number,input_count}).*...
  (ot{decoder_channel_number,input_count}.*(1-(tanh(ct{decoder_channel_number,input_count})).^2)),filters.Wdecoder{decoder_channel_number,decoder_kernel_number},'same').* ...
  ((ft{decoder_channel_number,input_count-1}.*it{decoder_channel_number,input_count-1}.*(1-ct{decoder_channel_number,input_count-1}.^2))+...
  it{decoder_channel_number,input_count}.*(1-ct{decoder_channel_number,input_count}.^2));
end
end
end

end
end
% test
[row_number_test,column_number_test] = size(Test_inputs{1,1});
xt_test = Test_inputs;
for i=1:layer_number
Ht_prev_test{i} = zeros(row_number_test,column_number_test);
ct_prev_test{i} = zeros(row_number_test,column_number_test);
end
input_number_test = size(Test_inputs,2);
for input_count_test = 1:input_number_test
% Encoder
for i=1:layer_number
xt_Wxi_test = zeros(row_number_test,column_number_test);
xt_Wxf_test = zeros(row_number_test,column_number_test);
xt_Wxc_test = zeros(row_number_test,column_number_test);
xt_Wxo_test = zeros(row_number_test,column_number_test);
for channel_count = 1:channel_number
    xt_Wxi_test = xt_Wxi_test+conv2(xt_test{channel_count,input_count_test},Wxi{channel_count,i},'same');
    xt_Wxf_test = xt_Wxf_test+conv2(xt_test{channel_count,input_count_test},Wxf{channel_count,i},'same');
    xt_Wxc_test = xt_Wxc_test+conv2(xt_test{channel_count,input_count_test},Wxc{channel_count,i},'same');
    xt_Wxo_test = xt_Wxo_test+conv2(xt_test{channel_count,input_count_test},Wxo{channel_count,i},'same');
end
it_test{i,input_count_test} = logsig(xt_Wxi_test+conv2(Ht_prev_test{i},Whi{i},'same')+conv2(ct_prev_test{i},Wci{i},'same')+bi{i});
ft_test{i,input_count_test} = logsig(xt_Wxf_test+conv2(Ht_prev_test{i},Whf{i},'same')+conv2(ct_prev_test{i},Wcf{i},'same')+bf{i});
chatt_test{i,input_count_test} = sigmoid_g(xt_Wxc_test+conv2(Ht_prev_test{i},Whc{i},'same')+bc{i});
ct_test{i,input_count_test} = ft_test{i,input_count_test}.* ct_prev_test{i}+it_test{i,input_count_test}.*chatt_test{i,input_count_test};
ot_test{i,input_count_test} = logsig(xt_Wxo_test+conv2(Ht_prev_test{i},Who{i},'same')+conv2(ct_prev_test{i},Wco{i},'same')+bo{i});
Ht_test{i,input_count_test} = ot_test{i,input_count_test}.*sigmoid_h(ct_test{i,input_count_test});
Ht_prev_test{i} = Ht_test{i,input_count_test};
ct_prev_test{i} = ct_test{i,input_count_test};
end
end