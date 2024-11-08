def compute():
    result = 0
    for i in range(100000000):
        result += i * i
    return result

num = compute()
print(num)
