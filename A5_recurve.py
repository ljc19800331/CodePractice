
'''
Recursive and Hanoir problems

'''

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