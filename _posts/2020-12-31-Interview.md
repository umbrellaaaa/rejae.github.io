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

## 五道组合题
337. 按摩师收益
```python
class Solution:
    def massage(self, nums: List[int]) -> int:

        # 递推： f(n)= f(n-2)+a[i] or f(n-1)
        # if len(nums)==0:
        #     return 0
        # result = 0
        # def recur(result,n):
        #     if n == 2: return max(nums[0]+nums[2],nums[1])
        #     if n == 1: return max(nums[0],nums[1])
        #     if n == 0: return nums[0]

        #     result += max(recur(result,n-2)+nums[n],recur(result,n-1))

        #     return result

        # result  = recur(result,len(nums)-1)
        # return result

        #将递推改为迭代
        n = len(nums)
        if n == 0 : return 0  

        if n == 1: return nums[0]
        if n == 2: return max(nums[0],nums[1])
        if n == 3: return max(nums[0]+nums[2],nums[1])
        f0 = nums[0]
        f1 = max(nums[0],nums[1])
        f2 = max(nums[0]+nums[2],nums[1])

        a = f1
        b = f2
        
        for i in range(3,len(nums)):

            c = max(a+nums[i],b)
            a = b
            b = c
        print(c)
        return c
```

46. 全排列: 从第一个位置起，每一个与首位交换，生成包括原顺序在内的共n个nums，把头固定，从第二个位置按此逻辑进行。注意返回的时候，递归无法返回引用，要么用一个1*1的数组解决。

```python
class Solution(object):
    def permute(self, nums):

        n = len(nums)
        # 递归，回溯，状态撤销

        def trans_back(first = 0):#传入的是第i个将发生交换的位置,第一个默认为1
            #当first到尾部的时候，返回排列
            if first == n:
                output.append(nums[:]) #由于nums在递归过程中不断变化，[:]表示复制
            # 对于任何一个位置，发生交换次数为当前位置到末尾
            for i in range(first,n):
                # 注意到需要所有的排列，第一个位置也是一个选项，所以与本身做交换也方便
                temp = nums[first]
                nums[first] = nums[i]
                nums[i] = temp
                trans_back(first+1)
                #撤销交换
                temp = nums[first]
                nums[first] = nums[i]
                nums[i] = temp
        
        output = []
        trans_back()
        return output
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

945. 使数组唯一的最小增量
给定整数数组 A，每次 move 操作将会选择任意 A[i]，并将其递增 1。

返回使 A 中的每个值都是唯一的最少操作次数。

示例 1:

输入：[1,2,2]
输出：1
解释：经过一次 move 操作，数组将变为 [1, 2, 3]。

思考：数组问题，将其中的一部分重复元素，每次加一，以实现数组无重复元素，返回最小加一的总次数

问题转化：数组问题，先将其排序，然后判断i-1位置与i位置元素的关系。最开始我使用的==比较，后来发现，当1,1,2,2,4出现的时候，先变成了1，2,2,2,4，然后变成了1,2,3,2,4，后面的一个2与3不满足这种关系，所以正确的方式应该用A[i-1]>=A[i]比较：

```
class Solution(object):
    def minIncrementForUnique(self, A):
        A = sorted(A)
        count = 0
        for i in range(1,len(A)):
            #print('i=',i,'A=',A)
            while A[i-1]>=A[i]:
                temp_Ai = A[i]
                A[i]=A[i-1]+1
                count+=A[i-1]-temp_Ai+1
        return count



       #hash 线性探测，路径压缩
        arr = [-1]*80000
        b=0
        def find_pos(index):
            # index 既是下标，又是值。进来后当做了下标
            b = arr[index] # 我们将数组中的值，当做hash索引，查看这个值对应的位置是否为空
            if b==-1: # 如果为空，返回这个值，后面计算做差是0 个move
                arr[index]=index
                b = index
                return b
            
            b = find_pos(b+1) #对应槽位已有元素，向后搜索。直到搜索到了空位
            arr[index]=b  # 将路径压缩：将
            return b
        move = 0
        for item in A:
            b = find_pos(item) #传进来的是值
            move += b-item
        
        return move
```





## 自报家门
本科专业 软件工程，硕士专业 计算机技术； 研究方向是自然语言处理。

研一开始从机器学习入手，然后慢慢过渡到深度学习，后来接触到了自然语言处理，并且身处于NLP高速发展的时期，对NLP很感兴趣并选择了这个方向为自己研究的方向。



个人的优点是 有较强的时间观念，较强的执行力，较强的探索学习能力，很强的适应能力

缺点要谈的话： 缺少从0到1的项目经验。



# 实战面试
-----------------------
时间限制：C/C++语言 1000MS；其他语言 3000MS
内存限制：C/C++语言 65536KB；其他语言 589824KB
题目描述：
首先给出你一个整数，可能为正也可能为负，这个数字中仅包含数字1-9，现在定义一个1-9的置换，即指定将整数中的某个数字按顺序变换为另一个数字，请你输出变换以后的数字是多少。

输入
 输入第一行包含一个整数a。(-10^1000<=a<=10^1000)

 输入第二行包含9个以空格隔开的整数a_i , 第i个整数表示将数字i替换为数字a_i。(1<=a_i<=9)

输出
请你输出数字变换之后的结果。


样例输入
1234567
9 8 7 6 5 4 3 2 1
样例输出
9876543

-----------------------
同心圆
时间限制：C/C++语言 1000MS；其他语言 3000MS
内存限制：C/C++语言 65536KB；其他语言 589824KB
题目描述：
有这么一个奇怪的符号：在一张正方形的纸上，有许多不同半径的圆。他们的圆心都在正方形的重心上（正方形的重心位于正方形两条对角线的交叉点上）。

最大的圆的外面部分是白色的。最外层的圆环被涂成了黑色，接下来第二大的圆环被涂层白色，接下来第三大的圆环被涂层黑色。以此类推，例如下图。

现在，给你这些圆的半径，请问黑色部分的面积是多少？精确到小数点后5位输出（四舍五入）。



输入
输入包含两行。第一行一个整数n，表示圆的个数。

接下来n个整数，描述每个圆的半径ri。数据保证没有两个圆的半径是一样的。(1<=n<=100 , ri<=1000)

输出
输出包含一个数，代表黑色部分的面积。


样例输入
5
1 2 3 4 5
样例输出
47.12389

提示
样例解释：
一共有5个圆(环)，其中最大的，第三大的，以及最小的圆环被染成了黑色。注意，最小的圆环已经退化为一个圆了。

---------------------------------------

子序列计数
时间限制：C/C++语言 3000MS；其他语言 5000MS
内存限制：C/C++语言 131072KB；其他语言 655360KB
题目描述：
一个序列是有趣的需要满足：当且仅当这个序列的每一个元素ai 满足 i 整除ai 。

现在给定一个长度为n的数组，问这个数组有多少个非空的子序列是有趣的，由于答案可能比较大，只需要输出在模998244353意义下答案的就行了。

一个长度为n的数组的非空子序列定义为从这个数组中移除至多n-1个元素后剩下的元素有序按照原顺序形成的数组。比如说对于数组[3,2,1]，它的非空子序列有[3],[2],[1],[3,2],[3,1],[2,1],[3,2,1]。

输入
第一行一个整数n表示序列的长度。(1<=n<=1e5)

第二行n个整数表示给定的序列。(1<=ai<=1e6)

输出
输出一个数表示有趣的子序列的个数。


样例输入
2
3 1
样例输出
2

---------------------------------
小仓的射击练习3
时间限制：C/C++语言 1000MS；其他语言 3000MS
内存限制：C/C++语言 65536KB；其他语言 589824KB
题目描述：
小仓酷爱射击运动。今天的小仓会进行N轮射击，已知每次击中靶心的概率为p/q，每当小仓击中靶心，如果是连续k次击中，那么小仓会获得a[k]的得分。小仓希望知道最后她的期望得分是多少。

输入
第一行三个数N，p，q代表射击轮数以及击中靶心概率。

第二行N个数a[i]，第i个数为a[i]。

1<=N<=100000

0<=a[i]<998244353

1<=p,q<998244353

输出
一个数表示期望得分。

不难证明答案有唯一的最简分数表示，若答案的最简表示为A/B，请输出A*B-1(mod 998244353)，B-1表示B在模998244353意义下的逆元，满足B-1*B≡1(mod 998244353)。又已知B-1≡B998244351（mod 998244353），故仅需输出A*B998244351(mod 998244353)。（ “≡”为数论中表示同余的符号）


样例输入
3 1 2
8 8 8
样例输出
12

提示
样例解释
三组射击会等概率地出现8种情况（000:0分 001:8分 010:8分 011:16分 100:8分 101:16分 110:16分 111:24分， 0表示偏离， 1表示击中）。故最后期望得分为（0+8+8+16+8+16+16+24）/8=96/8=12/1=12*1^(998244351)=12（mod 998244353）。


-------------------------------

套娃前缀和
时间限制：C/C++语言 1000MS；其他语言 3000MS
内存限制：C/C++语言 65536KB；其他语言 589824KB
题目描述：
套娃最近很流行，小美想知道前缀和是否也可以进行套娃操作。



小美现在想知道能否快速求解

 

输入
第一行两个数N，K。

第二行N个数，第 i 个数为a[i]。

1<=N<=5000 ；1<=K<998244353；0<=a[i]<998244353

        

输出
一个数表示sum[K][N]。


样例输入
4 3
1 0 0 0
样例输出
10

提示
样例解释，需要三次求和：
第一次：sum[1][1] = sum[0][1] = 1, sum[1][2] = sum[0][1] + sum[0][2] = 1, sum[1][3] = sum[0][1] + sum[0][2] + sum[0][3] = 1, sum[1][4] = sum[0][1] + sum[0][2] + sum[0][3] + sum[0][4] = 1. sum[1][] = { 1, 1, 1, 1 };
第二次：sum[2][] = { 1, 2, 3, 4 }.
第三次：sum[3][] = { 1, 3, 6, 10 }.
故sum[K][N] = sum[3][4] = 10



# 输入输出格式问题：
python的输入：
```python
input() #输入一行数据
list(map(int ,input().split())) #将一行数据通过map改变为int类型，然后转成list

```






#现给定任意正整数 n，请寻找并输出最小的正整数 m（m>9），使得 m 的各位（个位、十位、百位 ... ...）之乘积等于n，若不存在则输出 -1。
```python
def solution( n ):
    # write code here
    coins= [9,8,7,6,5,4,3,2]
    ele_list = []
    temp=[n]
    def helper(temp):
        print(temp[0])
        if temp[0]==1:
            return True

        for i in coins:
            if temp[0]%i==0 and temp[0]>=i:
                ele_list.append(str(i))
                print(i)
                temp[0]=int(temp[0]/i)
                return helper(temp)
            continue
            
        if temp[0]>10:
            return False

    if helper(temp):
        ele_list = ele_list[::-1]
        return ''.join(ele_list)
    return -1
```

```python
def solution( n ):
    # write code here
    # 1, 2,3,  4,5,6  7,8,9,10
    arr = []
    count = 1
    iner_count=0
    def get_ans(nn,n):
        print('mm=',nn)
        ans = 0
        for i in range(1,nn+1):
            ans+=i**2

        ans+=(n-((1+nn)*nn)/2)*(nn+1)
        return int(ans)
            
    for i in range(1,99999999):
        temp_area=[]
        for j in range(count):
            iner_count+=1
            if iner_count>n:
                return get_ans(len(arr[-1]),n)
            temp_area.append(count)
        count+=1
        arr.append(temp_area)
```






#输入m=1，n=2，表示最少1个键，最多2个键，符合要求的键数是1个键和2个键，其中1个键的有效模式有9种，两个键的有效模式有56种，
#所以最终有效模式总数是9+56=65种，最终输出65。
```python
def solution( m , n ):
    # A计算一种*4，B计算一种*4 + C
    A=[1,3,7,9]
    B=[2,4,6,8]
    C=5
    arr=[9]
    used=[]
    # 对于A中元素
    ans=[0]
    if m==1:
        return 9
    def get_m(used,m,count):
        if m==0:
            return
        for i in range(1,10): 
            if i not in used or used is None:
                used.append(i)
                element = used[-1]
                if len(used)>1:
                    if element in A:
                        ans[0]+=5-count+1
                    elif element in B:
                        ans[0]+=7-count+1
                    else:
                        ans[0]+=1
                get_m(used,m-1,count+1)
                print('used.pop',used)
                used.pop(-1)
                count-=1
    
    get_m([],m,1)
    return ans[0]
        
    
    
solution( 2 , 2 )
```  