# 跳台阶：febonacci F0 =0, F1 = 1, Fn= F(n-1) + F(n-2)
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

