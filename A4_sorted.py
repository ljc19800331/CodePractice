

'''
1. The usage of dictionary
2. The usage of range
3. The usage of list
4. The usage of loop (reverse or non-reverse)

reference: https://leetcode-cn.com/problems/sort-an-array/solution/python-shi-xian-de-shi-da-jing-dian-pai-xu-suan-fa/

'''

arr = [3, 2, 1, 4, 5, 6, 10, 9, 8]

# Selection sort -- similar to brute force searching
# choose a item and compare this item with all the other items and put at the end position
# every time find the maximum values in the remaining list
# Find the min one and the second min one and the third min one --
n = len(arr)
for i in range(n):
    for j in range(i, n):
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
print("arr = ", arr)

# bubble sort
# main idea: compare both and finally remove the greatest item to the end
for i in range(n):
    for j in range(n - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
print("arr = ", arr)

# Insertion Sort
# choose a item and insert to the previous sorted item list
for i in range(1, n):
    for j in range(i, -1, -1):
        if arr[i] < arr[j]:
            arr[j], arr[i] = arr[i], arr[j]
        else:
            i = j
print("arr = ", arr)





























