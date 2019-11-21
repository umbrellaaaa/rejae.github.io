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

## 树结构
1. 二叉树的构建



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
```python
# 1.sum two
class Solution(object):

    def sum_two(self, arr, target):
        m_dict = {}

        for index, num in enumerate(arr):
            if target - num in m_dict:
                return [m_dict[target - num], index]
            m_dict[num] = index


# 2. add_two
# 给定两个费控链表来表示两个非负整数。位数按照逆序方式存储，它们的每个节点只存储单个数字。将两个数相加返回一个新的链表
# 342+465 = 807    2-4-3   5-6-4  考虑进位
class ListNode(object):
    def __init__(self, val):
        self.val = val
        self.next = None


class Solution(object):

    def add_two(self, l1, l2):
        p = dumy_node = ListNode(-1)
        carry = 0

        while l1 and l2:
            carry = (l1.val + l2.val)/10
            p.next = ListNode((l1.val + l2.val + carry)%10)
            l1 = l1.next
            l2 = l2.next
            p = p.next
        result = l1 or l2
        while result:
            p.next = ListNode((carry+result.val)%10)
            carry = (carry+result.val)/10
            p = p.next
            result = result.next
        if carry:
            p.next = ListNode(1)

        return dumy_node.next

    def short_version(self,l1,l2):
        p = dumy_node = ListNode(-1)
        carry = 0
        while l1 or l2 or carry:
            var = (l1 and l1.val or 0) + (l2 and l2.val or 0) +carry
            carry = var / 10
            p.next = ListNode(var%10)
            l1 = l1 and l1.next
            l2 = l2 and l2.next
            p = p.next

        return dumy_node.next
```



```python
# 最长子串长度
# 3. max sub-string of no repeat charactor
class Solution(object):

    def length_of_longest_substring(self, s):
        ans = 0
        start = 0
        d = {}
        for i,v in enumerate(s):

            if v in d:
                start = max(start,d[v]+1)
            d[v] = i
            ans = max(ans, i-start+1)
        return ans
```

