

'''
Minimum distances in two strings implementation in Python
reference: https://blog.csdn.net/dongrixinyu/article/details/78771600
'''

# This shows the solution

strs = ['*','3','*', '5', '10', '9', '7', '1', '*']
str1 = '*'
str2 = '9'

list1 = []
list2 = []

def Func():

    # con1:
    if (str1 not in strs) or (str2 not in strs):
        return 1

    # con2:
    if str1 == str2:
        return 0

    # con3:
    for idx, item in enumerate(strs):

        # record the position of the str1 and str2
        if item == str1:
            list1.append(idx)
        if item == str2:
            list2.append(idx)

    # con4
    res_min = 1000
    temp = 0
    for i in list1:
        for j in list2:
            dis = abs(i - j)
            if dis <= res_min:
                res_min = dis

    print("list1 = ", list1)
    print("list2 = ", list2)
    print("res_min = ", res_min)

    # improvements -- only need con3 with two loops
    # for i in xrange(0, len(strs)):
    #     if str1 == strs[i]:
    #         pos1 = i
    #         for j in xrange(0, len(strs)):
    #             if str2 == strs[j]:
    #                 pos2 = j
    #             dist = abs(pos1 - pos2)
    #             if dist < min:
    #                 min = dist

Func()

