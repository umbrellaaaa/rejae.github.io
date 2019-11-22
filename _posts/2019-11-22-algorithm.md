---
layout:     post
title:      algorithm
subtitle:   yi题
date:       2019-11-22
author:     RJ
header-img: 
catalog: true
tags:
    - Algorithm

---
<p id = "build"></p>
---

## 构造链表
```python
# 构造链表:
# 定义节点类； 链表初始化；链表逆序； 增删改查


class Node(object):

    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList(object):

    def __init__(self):
        self.head = None
        self.length = 0

    def init_linked_list(self, data):
        if data:
            self.head = Node(data[0])
            self.length = len(data)
        else:
            return None

        p = self.head
        for value in data[1:]:
            p.next = Node(value)
            p = p.next

    def print_linked_list(self):
        if not self.head:
            return None

        p = self.head
        while p:
            print(p.value)
            p = p.next

    def reverse(self):
        pre = None
        while self.head and self.head.next:
            next = self.head.next
            self.head.next = pre
            pre = self.head
            self.head = next
        self.head = pre
        self.print_linked_list()
        return pre

    def add(self, index, value):  # 按1序列
        p = self.head
        if index == 1:
            self.head = Node(value)
            self.head.next = p
            self.print_linked_list()
            return None
        elif index > self.length:
            for i in range(self.length - 2):
                p = p.next
            p.next = Node(value)
            self.print_linked_list()
            return None
        else:
            for i in range(index - 2):  # 如果是3，那么只循环一次，到达被插节点的前一个节点
                p = p.next
        temp_node = p.next
        p.next = Node(value)
        p.next.next = temp_node

        self.print_linked_list()

    def delete(self, value):
        p = self.head
        if p.value == value:
            self.head = self.head.next
        while p and p.next:
            pre = p
            if p.next.value == value:
                pre.next = p.next.next
            p = p.next
        self.print_linked_list()

    def change(self, index, value):
        p = self.head
        if index > self.length:
            return None
        for i in range(index):
            p = p.next

    def find(self, value):
        p = self.head
        while p:
            if p.value == value:
                print(True)
                return True
            p = p.next
        print(False)
        return False


linkedList = LinkedList()
linkedList.init_linked_list([1, 2, 3, 4, 5, 6])
# linkedList.print_linked_list()
# linkedList.reverse()
# linkedList.find(2)
linkedList.delete(2)
linkedList.add(10, 11)

```

## 快速排序

```python
# 双指针


def quick_sort(arr, left, right):
    if left < right:
        pivot = arr[left]
        i, j = left, right
        while i < j:
            while i < j and arr[j] >= pivot:
                j = j - 1
            while i < j and arr[i] <= pivot:
                i = i + 1
            if i < j:
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp

        arr[left] = arr[i]
        arr[i] = pivot
        quick_sort(arr, left, i - 1)
        quick_sort(arr, j + 1, right)

    return arr


arr = [3, 1, 5, 4, 7, 8, 2, 6]


# quick_sort(arr, 0, 7)
# print(arr)

#单指针
def quick_sort_single(arr, start, end):
    if start < end:
        pivot = arr[start]
        mark = start
        i, j = start, end


        for index in range(i + 1, end + 1):
            if arr[index] < pivot:
                mark = mark + 1

                temp = arr[mark]
                arr[mark] = arr[index]
                arr[index] = temp

        arr[start] = arr[mark]
        arr[mark] = pivot
        quick_sort_single(arr, start, mark - 1)
        quick_sort_single(arr, mark + 1, end)


quick_sort_single(arr, 0, 7)
print(arr)
