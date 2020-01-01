---
layout:     post
title:      workday12
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


### 5. 中等 最长回文子串
首先区别 子串 与 子序列

input = 'abacabaaa'

output = abacaba


len= 1 return s

len==2 if s[0]==s[1] return s  else return None

定义数组： dp[i][j] 为从i位置到j位置的回文字符串

定义关系： dp[i] == dp[j]; if j-i  >1: i++,j-- ; dp[i]==dp[j]   j-i+1为回文长度

定义初值： dp[0]=0, dp[i]=i