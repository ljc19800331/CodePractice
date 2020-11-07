
'''
python2:
python3:

1. dictionary
2. range(0,i) + range(i, -1, -1)
3. pointer and index

'''

# Factorial Problem
def factorial(n):

    if n == 1:
        return 1
    if n > 1:
        return n * factorial(n - 1)

n = 3
print("res = ", factorial(n))

# Hanoi Tower Problem -- this is a very complicated graph-tree problem
def move(n, a, buffer, c):

    if(n == 1):
        print(a, "->", c)
        return

    # This code shows a very abstract concept for this problem -- pending

    # First move n-1 items from a to c through buffer
    move(n - 1, a, c, buffer)

    # Second move the item from a to c (the bottom object)
    move(1, a, buffer, c)

    # Finally, move the item from b to c (the bottom object) -- repeat the similar cases
    move(n - 1, buffer, a, c)

move(3, "a", "b", "c")

# Fib list problem
arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

def Fib(n):

    if n == 0:
        return 0

    if n == 1:
        return Fib(n - 1) + 1

    if n > 1:
        return Fib(n - 1) + Fib(n - 2)

n = 10

list_fib = []
for i in range(n):
    list_fib.append(Fib(i))

print("list_fib = ", list_fib)











