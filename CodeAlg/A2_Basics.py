
'''
Python: adjust the strings to the left and right
reference: https://blog.csdn.net/dongrixinyu/article/details/78771600

# algorithm:
1. find the list for *
2. find the list for number
3. form a new list for all the numbers

1. if not allowed to copy list -- use double pointers
2. if allowed -- use two lists of strings for this problems.

'''

strs = ['*', '1', '2', '*', '4', '5', '*']
str1 = []
str2 = []
strs_res = []

for item in strs:

    if item == '*':
        str1.append(item)
    if item != '*':
        str2.append(item)

idx_1 = 0       # count for the first list -- *
idx_2 = 0       # count for the second list -- number
idx_global = 0  # index for the final list

for i in range(len(strs)):

    # list1
    # print(len(str1))
    Ncheck = (len(str1) - 1)
    # print(Ncheck)
    # print(idx_1 <= Ncheck)
    if (idx_1 <= (len(str1) - 1)):
        strs_res.append('*')
        idx_1 += 1
        continue

    # list2
    if (idx_2 <= (len(str2) - 1)):
        strs_res.append(str2[idx_2])
        idx_2 += 1

# improvement -- separate the items into two groups
# def replace_stars(self, str_list):  # 将所有*号移动到数组的左侧
#     j = len(str_list) - 1
#     for i in xrange(len(str_list) - 1, -1, -1):
#         if str_list[i] != '*':
#             str_list[j] = str_list[i]
#             j -= 1
#     for i in xrange(0, j + 1):
#         str_list[i] = '*'
#     return str_list

print("strs = ", strs)
print("str1 = ", str1)
print("str2 = ", str2)
print("strs_res = ", strs_res)















