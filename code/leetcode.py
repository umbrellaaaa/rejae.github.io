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