

# Math Module:

# Find the solutions for the cases:

# def greatest_common_divisor_1(self, num1, num2):
#     '''
#     数值计算寻找最大公约数，给定两个整数，计算其最大公约数，时间复杂度为 o(min(num1,num2))，取余运算复杂度高
#     '''
#     gbc = 1
#     for i in xrange(2, min(num1, num2) + 1):
#         if num2 % i == 0 and num1 % i == 0:
#             gbc = i
#     return gbc

# Fibonacci sequence

# n = 10
#
#
# class Fibonacci(object):
#
#     def __init__(self):
#         a = 1
#
#     def FibFunc(self, n):
#
#         if n == 1:
#             return 0
#
#         if n == 2:
#             return 1
#
#         return self.FibFunc(n - 1) + self.FibFunc(n - 2)
#
# test = Fibonacci()
# print(test.FibFunc(n = 10))

# Eight Queens (pending and im cases)

# Make a mesh grid
N = 8
data_mesh = []
for i in range(N):
    data_line = []
    for j in range(N):
        data_line.append(j)
    data_mesh.append(data_line)

print("data_mesh = ", data_mesh)

# For each new point -- update the new data
for i in range(N):
    # index for the targets
    for j in range(N):
        # index for the mesh
        for k in range(N):
            # index for the mesh
            a = 1
            # sum of row
            # for i1 in range(N):

            # sum of col
            # for i2 in range(N):

            # sum of diagonal -- more
            # for i3 in range(N):

