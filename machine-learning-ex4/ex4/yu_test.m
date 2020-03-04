a = [1;2;3;4;6];
b = zeros(size(a,1),10);
for i = 1:size(a,1)
    b(i,:) = 1:10;
    b(i,:)=(b(i,:) == a(i));
end
b
    