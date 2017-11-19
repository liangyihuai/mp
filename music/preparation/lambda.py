from functools import reduce;
foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print(list(filter(lambda x: x % 3 == 0, foo)))
print([x for x in foo if x%3 == 0]);
print(list(map(lambda x: x * 2 + 10, foo)));
print([x * 2 + 10 for x in foo]);
print(reduce(lambda x, y: x + y, foo));

fs = [i+1 for i in range(5)];
print(fs);
