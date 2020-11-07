
'''
1. Dynamic Programming
2. Coin Change Problem
'''

# Coin Change problem
# res = 3
# res_list = [1, 5, 5] or [5, 5, 1]
# Goal: Find the minimum cases and all the possible combinations

def dp(n):

    # input: The current sum of number
    res_final = 100   # initial number of cases

    # Some stopping criteria first
    # base case and this is out of the loop
    # transition probabiltiy: n = 0 -> return 0 + n = 1 -> return -1 else -> return continue (this equals to cancel the list)
    if n == 0:
        # return 0 since this is already being counted by in(res_final, 1 + res_temp)
        return 0

    # cancel case where the item is not optimal candidate
    if n < 0:
        # cancel this list -- flag = -1
        return -1

    # Loop over the choices of coins
    for coin in coins:

        # if n - coin > 0:
        res = dp(n - coin)
        res_temp = dp(n - coin)     # a new temporary value

        if res == -1:
            # cancel this list
            # flag = -1 -- remove this list
            continue

        # Find the optimal res at the current branches
        # here, the 1 shows the number from the last layer
        # The idea of single loop recursive operation
        res_final = min(res_final, 1 + res_temp)

        continue

    return res_final if res_final != 100 else -1

# The main function
coins = [1, 2, 5]
n = 11
res = dp(n)
print("The minimum number of coins is ", res)





