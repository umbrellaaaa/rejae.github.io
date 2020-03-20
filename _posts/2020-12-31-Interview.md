---
layout:     post
title:      Interview
subtitle:   
date:       2020-12-31
author:     RJ
header-img: 
catalog: true
tags:
    - Job


---
<p id = "build"></p>
---

## 动态规划

[告别动态规划](https://zhuanlan.zhihu.com/p/91582909)

动态规划（英语：Dynamic programming，简称 DP）是一种在数学、管理科学、计算机科学、经济学和生物信息学中使用的，通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。

动态规划常常适用于有重叠子问题和最优子结构性质的问题，动态规划方法所耗时间往往远少于朴素解法。

动态规划背后的基本思想非常简单。大致上，若要解一个给定问题，我们需要解其不同部分（即子问题），再根据子问题的解以得出原问题的解。动态规划往往用于优化递归问题，例如斐波那契数列，如果运用递归的方式来求解会重复计算很多相同的子问题，利用动态规划的思想可以减少计算量。

通常许多子问题非常相似，为此动态规划法试图仅仅解决每个子问题一次，具有天然剪枝的功能，从而减少计算量：一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。

KEY WORD: 子问题， 记忆查表， 剪枝， 优化递归

解决问题三大步：
1. 定义数组元素的含义
2. 找出数组元素之间的关系式
3. 定义初始值
----
1. example: 跳台阶

f(n)=f(n-1)+f(n-2)

f(1)=1, f(2)=2, f(3)=f(1)+f(2)
```python
def func(n):

    if n ==2:
        return 2
    if n ==1:
        return 1
    return f(n-1)+f(n-2)
```

```java
int f( int n ){
    if(n <= 1)
    return n;
    // 先创建一个数组来保存历史数据
    int[] dp = new int[n+1];
    // 给出初始值
    dp[0] = 0;
    dp[1] = 1;
    // 通过关系式来计算出 dp[n]
    for(int i = 2; i <= n; i++){
        dp[i] = dp[i-1] + dp[i-2];
    }
    // 把最终结果返回
    return dp[n];
}

```

2. example:一个机器人位于一个 m x n 网格的左上角, 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角

定义数组： dp[i][j]为存储从左上角到i,j位置的路径方式

定义关系： dp[i][j] = dp[i-1][j] + dp[i][j-1]

定义初值： dp[0][:n-1]=1,  dp[:n-1][0]=1

```java
public static int uniquePaths(int m, int n) {
    if (m <= 0 || n <= 0) {
        return 0;
    }

    int[][] dp = new int[m][n]; // 
    // 初始化
    for(int i = 0; i < m; i++){
      dp[i][0] = 1;
    }
    for(int i = 0; i < n; i++){
      dp[0][i] = 1;
    }
        // 推导出 dp[m-1][n-1]
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}
```


5. 中等 最长回文子串
首先区别 子串 与 子序列

input = 'abacabaaa'

output = abacaba


len= 1 return s

len==2 if s[0]==s[1] return s  else return None

定义数组： dp[i][j] 为从i位置到j位置的回文字符串

定义关系： dp[i] == dp[j]; if j-i  >1: i++,j-- ; dp[i]==dp[j]   j-i+1为回文长度

定义初值： dp[0]=0, dp[i]=i

由 s, dp[len(s)], dp[i]存储s[i]为中心，两边对称的最长回文。

42. 接雨水问题

思路： 站在点 i 看 Min(max(A[:i+1]) 和 max(A[i:])) 该位置的接水量，就是这个值减去当前柱子高度。
```python

public class Solution:


    def max_area(arr0):
        result = 0
        if len(arr)<=2:
            return 0
        for i in range(1,len(arr)-1):
            result += min(max(arr[:i+1]),max(arr[i:])) - arr[i]
        
        return result


class Solution:
    def trap(self, height: List[int]) -> int:

        n = len(height)
        if n <2: return 0
        liquid = height[:]
        for i in range(1,n-1):
            liquid[i]=min(max(height[:i+1]),max(height[i:]))
        return sum(liquid)-sum(height)
```
step1: 分析变量

数组： arr

变量： i

step2:判断临界条件

if n<=2:
    return 0

step3:


72. 编辑距离
给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:

```
思考： A-->>B

A[:i-1] B[:j-1]

A[:i-1] B[:j]

A[:i] B[:j-1]
```python
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n1, n2 = len(word1), len(word2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        dp[0][0] = 0
        for i in range(1, n1 + 1):
            dp[i][0] = i
        for j in range(1, n2 + 1):
            dp[0][j] = j
            
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[n1][n2]
```

## 链表问题

206. 反转链表
```
class Solution:

    
    def reverseList(self, head: ListNode) -> ListNode:
        
        
        prev = None
        cur =head
        while cur:
            nextTemp = cur.next # 存储下一节点
            cur.next = prev # 断开反向重连
            prev = cur # prev后移一位
            cur = nextTemp #Cur 后移一位
        
        return prev
```

94. 二叉树中序遍历

```python
class Solution(object):
	def inorderTraversal(self, root):
		"""
		:type root: TreeNode
		:rtype: List[int]
		"""
		res = []
		def dfs(root):
			if not root:
				return
			# 按照 左-打印-右的方式遍历	
			dfs(root.left)
			res.append(root.val)
			dfs(root.right)
		dfs(root)
		return res

```

### 用Java写算法

```java
public class TreeNode{
    int value;
    TreeNode left;
    TreeNode right;

    //构造方法
    TreeNode(int x){
        value = x
    }
}

class Solution {
    public List < Integer > inorderTraversal(TreeNode root) {
        List < Integer > res = new ArrayList < > ();
        helper(root, res);
        return res;
    }

    public void helper(TreeNode root, List < Integer > res) {
        if (root != null) { //两次判断，冗余；直接判断node为null后就return  ;
            if (root.left != null) {
                helper(root.left, res);
            }
            res.add(root.val);
            if (root.right != null) {
                helper(root.right, res);
            }
        }
    }
}
```

将python的list创建迁移： List < Integer > res = new ArrayList < > ();

基于栈的二叉树，中序遍历
```java
public class Solution {
    public List < Integer > inorderTraversal(TreeNode root) {
        List < Integer > res = new ArrayList < > ();
        Stack < TreeNode > stack = new Stack < > ();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            res.add(curr.val);
            curr = curr.right;
        }
        return res;
    }
}

```

15. 三数之和 ： 三指针

```python
class Solution:
    def threeSum(self, nums: [int]) -> [[int]]:
        nums.sort()
        res, k = [], 0
        for k in range(len(nums) - 2):
            if nums[k] > 0: break # 1. because of j > i > k.
            
            if k > 0 and nums[k] == nums[k - 1]: continue # 2. skip the same `nums[k]`.
            i, j = k + 1, len(nums) - 1

            while i < j: # 3. double pointer
                s = nums[k] + nums[i] + nums[j]
                if s < 0:
                    i += 1
                    while i < j and nums[i] == nums[i - 1]: i += 1
                elif s > 0:
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]: j -= 1
                else:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]: i += 1
                    while i < j and nums[j] == nums[j + 1]: j -= 1
        return res


```

111. 二叉树的最小深度

注意最小深度为从根节点到最近叶子节点的路径上的节点数

要求从根节点到子树，当左子树或右子树为空时，不符合要求
```python
class Solution(object):
    def minDepth(self, root):
        if root ==None:
            return 0
        if root.left==None and root.right!=None:
            return self.minDepth(root.right)+1
        if root.right==None and root.left!=None:
            return self.minDepth(root.left)+1

        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        return min(left,right)+1
```

104. 二叉树的最大深度

```python
class Solution(object):
    def maxDepth(self, root):
        if root ==None:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left,right)+1
```

二叉树所有节点数 与 叶子节点数 对比

```python
def func(root):
    if root ==None: return 0
    left = func(root.left)
    right = func(root.right)
    return left+right+1
```

```python
def func(root):
    if root == None: return 0
    if root.left ==None and root.right ==None:
        return 1
    return func(root.left)+func(root.right)
```

24. 两两交换链表节点

```python
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        ## 1. 终止条件
        if head ==None or head.next ==None:
            return head
        ## 2. 递归核心
        tail = head.next  #存
        head.next = self.swapPairs(tail.next) # 下一阶段联系建立
        tail.next = head # tail指向head
        ## 3. 返回值
        return tail
```

10. 正则表达式匹配

s 与 p , 我们遍历p:

```python
flag = True
C = [a-zA-Z]
trans = 0
for i in range(len(p)):
    if s[i]==p[i]:
        continue
    if p[i] in C and p[i]!=s[i] and flag:
        return False
    
    if p[i]=='.' and p[i+1] != '*':
        if p[i+1]==s[i]:
            trans = -1
    
    if p[i]=='.' and p[i+1]=='*':
        p[i+1]

    

```
11. 最大容积
```python
class Solution(object):
    def maxArea(self, height):
        result =0 
        i=0
        j=len(height)-1
        while i<j:
            result = max(min(height[i],height[j])*(j-i),result)
            if height[i]<height[j]:
                i+=1
            else:
                j-=1
        return result
```

21. 合并两个有序链表
递归解法
```python
class Solution:
    def mergeTwoLists(self, l1, l2):
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

```

70. 爬楼梯
```python
    def climbStairs(self, n):
        # # f(n) = f(n-1)+f(n-2)
        if n <=2:
            return n
        
        # return self.climbStairs(n-1)+self.climbStairs(n-2)
        f1 = 1
        f2 = 2
        f = 0
        for i in range(3,n+1):
            f  = f1+f2
            f1 = f2
            f2 = f
        return f
```


面试题：字符串压缩
```python
class Solution(object):
    def compressString(self, S):
        ## 终止条件
        if not S:
            return ""
        result = ''
        save_S = S
        while S:
            j=1
            temp= S[0]
            count =1
            while j<len(S) and S[0]==S[j]:
                count+=1
                j+=1
            result+=temp+str(count)
            S=S[count:]
        if len(result)>=len(save_S):
            result=save_S
        return result
```

169. 数组多数元素
```python
class Solution:
    def majorityElement(self, nums):
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get)

class Solution(object):
    def majorityElement(self, nums):
        m_dic = dict()
        for item in nums:
            if str(item) not in m_dic:
                m_dic[str(item)]=1
            else:
                m_dic[str(item)]+=1
        # 将字典按值排序，返回键，toint
        sorted_list = sorted(m_dic.items(),key=lambda x:x[1],reverse=True)
        #print(sorted_list)
        return int(sorted_list[0][0])
```

215. 数组中的第K个最大元素 (快排)

```python
class Solution(object):
    def findKthLargest(self, nums, k):
        start = 0
        end = len(nums)-1
        def quick_sort(arr,start,end):
            if start>=end:
                return arr
            index = parition(arr,start,end)
            quick_sort(arr,start,index-1)
            quick_sort(arr,index+1,end)
            return arr

        
        def parition(arr,start,end):
            pivot = arr[start]
            i,j=start,end
            while i<j :

                while i<j and arr[j]>pivot:
                    j-=1

                while i<j and arr[i]<=pivot:
                    i+=1
                if i<j:
                    temp = arr[i]
                    arr[i]=arr[j]
                    arr[j]=temp
            arr[start] = arr[i]
            arr[i] = pivot
            return i

        arr = quick_sort(nums,start,end)
        return arr[-k]


```

20. 有效的括号
```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''
```
栈，解决问题：
```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """

        # The stack to keep track of opening brackets.
        stack = []

        # Hash map for keeping track of mappings. This keeps the code very clean.
        # Also makes adding more types of parenthesis easier
        mapping = {")": "(", "}": "{", "]": "["}

        # For every bracket in the expression.
        for char in s:

            # If the character is an closing bracket
            if char in mapping:

                # Pop the topmost element from the stack, if it is non empty
                # Otherwise assign a dummy value of '#' to the top_element variable
                top_element = stack.pop() if stack else '#'

                # The mapping for the opening bracket in our hash and the top
                # element of the stack don't match, return False
                if mapping[char] != top_element:
                    return False
            else:
                # We have an opening bracket, simply push it onto the stack.
                stack.append(char)

        # In the end, if the stack is empty, then we have a valid expression.
        # The stack won't be empty for cases like ((()
        return not stack
```



33. 搜索旋转排序数组
二分查找，部分有序：
```python
class Solution(object):
    def search(self, nums, target):
        if not nums : return -1
        i,j = 0,len(nums)-1
        while i<=j:
            mid = (i+j)//2
            if nums[mid]==target: return mid

            if nums[i]<=nums[mid]:# left
                if nums[i]<=target<nums[mid]: j=mid-1
                else: i=mid+1

            else: # right
                if nums[mid]<target<=nums[j]: i=mid+1
                else: j=mid-1
                
        return -1
```

43. 字符串相乘
```
class Solution:
    def multiply(self, str1: str, str2: str) -> str:
        m_dict =dict()
        for i in range(10):
            m_dict[str(i)]=i

        result = 0
        for index_2,item_2 in enumerate(str2):
            mul_2 = max(10**(len(str2)-index_2-1),1)
            for index_1, item_1 in enumerate(str1):
                mul_1 = max(10**(len(str1)-index_1-1),1)
                result += m_dict[item_2]*mul_2*m_dict[item_1]*mul_1

        return str(result)
```

54. 螺旋矩阵
```python
class Solution(object):
    def spiralOrder(self, matrix):
        if not matrix: return []
        R, C = len(matrix), len(matrix[0])
        seen = [[False] * C for _ in matrix]
        ans = []
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        r = c = di = 0
        for _ in range(R * C):
            ans.append(matrix[r][c])
            seen[r][c] = True
            cr, cc = r + dr[di], c + dc[di]
            if 0 <= cr < R and 0 <= cc < C and not seen[cr][cc]:
                r, c = cr, cc
            else:
                di = (di + 1) % 4
                r, c = r + dr[di], c + dc[di]
        return ans
```

394. 字符串解码

s = "3[a]2[bc]", 返回 "aaabcbc".
```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, multi = [], "", 0
        for c in s:
            if c == '[':
                stack.append([multi, res])
                res, multi = "", 0
            elif c == ']':
                cur_multi, last_res = stack.pop()
                res = last_res + cur_multi * res
            elif '0' <= c <= '9':
                multi = multi * 10 + int(c)            
            else:
                res += c
        return res
```

13. 罗马数字转整数

 左边小于右边，减这个数，否则加这个数
```
class Solution:
    def romanToInt(self, s: str) -> int:
        result = 0
        a = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}   
        
        for i in range(len(s)):
            if i<len(s)-1 and  a[s[i]]<a[s[i+1]]:
                result -= a[s[i]]
            else:
                result+=a[s[i]]
        return result
```





























## 自报家门
本科专业 软件工程，硕士专业 计算机技术； 研究方向是自然语言处理。

研一开始从机器学习入手，然后慢慢过渡到深度学习，后来接触到了自然语言处理，并且身处于NLP高速发展的时期，对NLP很感兴趣并选择了这个方向为自己研究的方向。



个人的优点是 有较强的时间观念，较强的执行力，较强的探索学习能力，很强的适应能力

缺点要谈的话： 缺少从0到1的项目经验。


# 面试总结
一方面，宽度和深度的结合比较好，一方面，缺少从0到1的项目。

编程功底有待提升，不只是实现一些方法，更重要的是要掌握整个项目流程，要有从创建实现最后上线的能力。

之前看的论文要多总结，一些面试的常考点必须写进blog.

选定NLP的两个大方向，一直往前慢慢推进就好。比如文本分类，机器翻译。

1. 序列标注：分词/POS Tag/NER/语义标注
2. 分类任务：文本分类/情感计算
3. 句子关系判断：Entailment/QA/自然语言推理
4. 生成式任务：机器翻译/文本摘要

以上，序列标注要掌握，这是基础。

接下来要了解腾讯云工单的处理相关问题。计算一个值，而不是简单的分难易的多分类。

笔试：现在每天刷5道，每周三十道，希望自己能获得较大的提升。

