---
layout:     post
title:      Interview 
subtitle:   summary
date:       2019-11-8
author:     RJ
header-img: 
catalog: true
tags:
    - ML

---
<p id = "build"></p>
---

<h1>面试算法汇总</h1>

## 递归
1. 跳台阶问题： step = 1 or 2 ;  sum = n ; question: count? <br>
跳台阶：febonacci F0 =0, F1 = 1, Fn= F(n-1) + F(n-2)
```python
# 
def febonacci(n):

    if n < 0:
        raise Exception("n must>0")
    if n > 1:
        return febonacci(n - 1) + febonacci(n - 2)
    else:
        return n


print(febonacci(6))


def good_febonacci(n):
    if n <= 1:
        return n, 0

    else:
        (a, b) = good_febonacci(n - 1)
        return a + b, a


print(good_febonacci(4)[0])
print(good_febonacci(5)[0])
print(good_febonacci(6)[0])
```

## 排序
1. 快排：双指针和单指针

```python
# 快排 双指针：
def quick_sort(arr, left, right):
    if left < right:
        pivot = arr[left]
        i, j = left, right
        while i < j:
            while arr[j] >= pivot and i < j:
                j = j - 1
            while arr[i] <= pivot and i < j:
                i = i + 1

            if i < j:
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
        # 交换在外部
        arr[left] = arr[i]
        arr[i] = pivot
        quick_sort(arr, left, i - 1)
        quick_sort(arr, j + 1, right)
        return arr


# arr = [3, 1, 5, 4, 7, 8, 2, 6]
# quick_sort(arr, 0, 7)
# print(arr)


# 单指针
def quick_sort_single(arr, start, end):
    if start < end:
        pivot = arr[start]
        mark = start

        for index in range(start + 1, end + 1):
            if arr[index] < pivot:
                mark = mark + 1
                temp = arr[mark]
                arr[mark] = arr[index]
                arr[index] = temp
        arr[start] = arr[mark]
        arr[mark] = pivot
        quick_sort_single(arr, start, mark - 1)
        quick_sort_single(arr, mark + 1, end)


arr = [3, 1, 10, 5, 4, 7, 8, 2, 6]
quick_sort_single(arr, 0, 8)
print(arr)

```

## LEETCODE
1. sum_two: