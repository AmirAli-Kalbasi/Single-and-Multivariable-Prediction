function out = max_pool(input,size_kernel,stride)
[m,n] = size(input);
if rem(m,2)~=0
   m = m+1;
   input(m,1:n) = zeros(1,n);
end
if rem(n,2)~=0
   n = n+1;
   input(1:m,n) = zeros(m,1);
end
L = ceil(size_kernel/2);
m1  = length(L : stride: m-L);
n1 = length(L : stride : n-L);
out = zeros(m1,n1);
counter1 = 1;
for i = L+1 : stride: m-L+1
    counter2 = 1;
    for j = L+1 : stride : n-L+1
        temp_window = input(i-L:i+L-1,j-L:j+L-1);
        out(counter1,counter2) = max(max(temp_window));
        counter2 = counter2+ 1;
    end
    counter1 = counter1+1;
end



