function bias = initial_bias(input_size,layer_number)
lower_bound = -0.1;
upper_bound = 0.1;
for i=1:layer_number
bias.bi{i} = unifrnd(lower_bound,upper_bound,[input_size.w,input_size.l]);
bias.bf{i} = unifrnd(lower_bound,upper_bound,[input_size.w,input_size.l]);
bias.bc{i} = unifrnd(lower_bound,upper_bound,[input_size.w,input_size.l]);
bias.bo{i} = unifrnd(lower_bound,upper_bound,[input_size.w,input_size.l]);
bias.decoder{i} = unifrnd(lower_bound,upper_bound,[input_size.w,input_size.l]);
end