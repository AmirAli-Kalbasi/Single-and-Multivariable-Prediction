function filters = initial_kernel(kernel_number,kernel_width,kernel_length,channel_number)
lower_bound = -0.1;
upper_bound = 0.1;
for i=1:kernel_number
    for channel_count = 1:channel_number
    filters.Wxi{channel_count,i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    filters.Wxf{channel_count,i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    filters.Wxc{channel_count,i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    filters.Wxo{channel_count,i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    filters.Wdecoder{i,channel_count} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]); 
    % kernel number of encoder = channel number of decoder
    % channel number of input = kernel number of decoder
    end
    filters.Whi{i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    filters.Wci{i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);

    filters.Whf{i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    filters.Wcf{i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    
    filters.Whc{i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    
    filters.Who{i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
    filters.Wco{i} = unifrnd(lower_bound,upper_bound,[kernel_width,kernel_length]);
end
