

# Math Module:
# 1: prime number -- https://labuladong.gitbook.io/algo/gao-pin-mian-shi-xi-lie/4.1-shu-xue-yun-suan-ji-qiao/da-yin-su-shu
# 2. other cases -- pending

# Find the Prime Number
# First: Find all the possible prime from [2, sqrt(n)]
n = 20

def isprime(num):

    if num <= 3:
        return True

    for i in range(2, num):

        # if divide by some values -- not a prime
        if num % i == 0:
            return False

    # Else -- this is a prime number
    return True

# Goal: return all the numbers of the prime
def PrimeFromN(n):

    # record the list of res
    list_prime = [1] * n

    # Loop over n
    for i in range(2, n):
        print("isprime(i) = ", isprime(i), i)
        if isprime(i):
            # multiplity of i is not a prime number
            for j in range(2 * i, n, i):
                list_prime[j] = 0
    print("list_prime = ", list_prime)

    # Count the list of the prime numbers
    list_res = []
    for i in range(n):
        if list_prime[i] == 1:
            list_res.append(i)

    return list_res

list_res = PrimeFromN(n = 20)
print("The final result is ", list_res)

exit()

Ninit = 0
Ndelta = 3
Nmax = 10
for i in range(Ninit, Nmax, Ndelta):
    print(i)

# Find the solutions for the cases:

# def greatest_common_divisor_1(self, num1, num2):
#     '''
#     Find the common number
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

# # Make a mesh grid
# N = 8
# data_mesh = []
# for i in range(N):
#     data_line = []
#     for j in range(N):
#         data_line.append(j)
#     data_mesh.append(data_line)
#
# print("data_mesh = ", data_mesh)
#
# # For each new point -- update the new data
# for i in range(N):
#     # index for the targets
#     for j in range(N):
#         # index for the mesh
#         for k in range(N):
#             # index for the mesh
#             a = 1
#             # sum of row
#             # for i1 in range(N):
#
#             # sum of col
#             # for i2 in range(N):
#
#             # sum of diagonal -- more
#             # for i3 in range(N):

