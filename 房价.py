import matplotlib.pyplot as plt

year = [2004, 2005, 2006, 2007, 2008, 2009, 2010,
    2011, 2012, 2013, 2014, 2015, 2016, 2017,
    2018, 2019, 2020, 2021, 2022, 2023]#年份

prices = [4631.68, 4965.31, 6531.69, 8531.15, 8848, 9231, 11615,
    10725.24, 12602.58, 12974.45, 13864.56, 14987, 16641, 17243,
    20321.98, 23952, 26871, 31312, 28965, 30344.56]#房价

#设函数为y=mx+b,使用最小二乘法求出m,b

n = len(prices)
x_avg = sum(year) / n
y_avg = sum(prices) / n
x2 = sum(year[i]**2 for i in range(n)) / n
xy_avg = sum(year[i]*prices[i] for i in range(n)) / n

#求出m，b的值
m = (xy_avg - x_avg * y_avg) / (x2 - x_avg**2)#斜率
b = y_avg - m * x_avg#截距

Price = [year[i] * m + b for i in range(n)]

# 绘制拟合结果
plt.figure(figsize=(10, 6))
plt.plot(year, prices, 'o' ,label = 'data')
plt.plot(year, Price, label = 'fit')
plt.legend()
plt.xticks(year)
plt.xlabel('year')
plt.ylabel('prices')
plt.show()