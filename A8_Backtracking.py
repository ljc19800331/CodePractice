
'''
1. Backtracking Idea

Summary:
1. Before recursion: make a decision
2. After recursion: cancel the decision
3. Core idea: delete the choice
4. reference: https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-python-dai-ma-java-dai-ma-by-liweiw/

Sudo Code:
result = []
def backtrack(path, choices):
    if satisfy stopping conditions:
        result.add(path)
        return

    for choice in choices:
        make a choice
        backtrack(path, choices)
        cancel the choice
'''

# # Define the global path list
# global res
# res = []
#
# # Practice: Permutation Problem
# def PT(path, choices):
#
#     # Stopping criteria
#     if len(path) == 3 and len(choices) == 0:
#         res.append(path)
#         print("res = ", res)
#         return 0
#
#     path_old = path
#     choices_old = choices
#
#     for choice in choices:
#
#         # choose the items
#         choices_use = []
#         for item in choices:
#             choices_use.append(item)
#         choices_use.remove(choice)
#
#         # Backtracking
#         path.append(choice)
#         res_temp = PT(path, choices_use)
#
#         # Return to the original states
#         # problems -- this is adding one more layer
#         if res_temp == 0:
#             path = path_old
#             choices = choices_old
#             continue
#
#     return res
#
# path = []
# choices = [1, 2, 3]
# PT(path, choices)
# print("The result of the permutation is ", res)
# exit()

# The core of permutation:
# 1. 1->2 + 1->3 +

def permutations(arr, position, end):

    # input:
    # position: current position
    # end: end position

    # Stopping criteria
    if position == end:
        print("res = ", arr)
    else:
        for index in range(position, end):

            # make a choice
            print("current arr = ", arr)
            print("index, position = ", index, position)
            print("arr[index], arr[position] = ", arr[index], arr[position])
            print("arr[position], arr[index] = ", arr[position], arr[index])
            input("check")
            arr[index], arr[position] = arr[position], arr[index]

            # backtracking -- move deeper
            permutations(arr, position + 1, end)

            # return to the original item (remove item or list)
            # keep the original state and use for next iteration
            arr[index], arr[position] = arr[position], arr[index]

arr = [1, 2, 3]
permutations(arr, 0, len(arr))

# N queens problem -- backtracking practices
# goal implement the N queens problem with backtracking
# This is a abstract concept and should run the program with more practices














