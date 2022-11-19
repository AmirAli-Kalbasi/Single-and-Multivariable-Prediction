%% Normalized data
% data_Normalized = Normalized(data,method)
% methode.name = ...
%                (1) 'range' : Rescale range of data to [method.lower_bound,method.upper_bound]
%                              % required: methode.name, method.lower_bound, method.upper_bound
function data_Normalized = Normalized(data,method)
if strcmp(method.name,'range')
    min_data = min(data);
    max_data = max(data);
    for i_normalized = 1:size(data,1)
        for j_normalized = 1:size(data,2)
            data_Normalized1(i_normalized,j_normalized) = (data(i_normalized,j_normalized) - min_data)/(max_data - min_data);
        end
    end
    data_Normalized = (method.upper_bound-(method.lower_bound))*data_Normalized1 + method.lower_bound;
elseif strcmp(method.name,'zscore')
    mean_data = mean(data);
    std_data = std(data);
    for i_normalized = 1:size(data,1)
        for j_normalized = 1:size(data,2)
            data_Normalized1(i_normalized,j_normalized) = (data(i_normalized,j_normalized) - mean_data)/(std_data);
        end
    end 
end