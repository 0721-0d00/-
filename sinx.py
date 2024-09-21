import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(0, 2 * np.pi, 100)  # 在0到2π范围内生成100个点
y = np.sin(x)  # 对应的y值为sin(x)

# 对输入数据进行标准化 (均值为0，标准差为1)
x_mean = np.mean(x)
x_std = np.std(x)
x_standardized = (x - x_mean) / x_std

# 初始化参数
theta = np.random.randn(6)


# 定义均方误差损失函数
def compute_cost(x, y, theta):
    m = len(y)
    pd = theta[0] + theta[1] * x + theta[2] * (x ** 2) + theta[3] * (x ** 3) + theta[4] * (x ** 4) + theta[
        5] * (x ** 5)
    return (1 / (2 * m)) * np.sum((pd - y) ** 2)


# 梯度下降算法
def gradient_descent(x, y, theta, eva, time):
    m = len(y)
    cost = np.zeros(time)
    for i in range(time):
        pd = theta[0] + theta[1] * x + theta[2] * (x ** 2) + theta[3] * (x ** 3) + theta[4] * (x ** 4) + theta[
            5] * (x ** 5)

        # 计算每个 theta 的梯度
        theta_0_grad = (1 / m) * np.sum(pd - y)
        theta_1_grad = (1 / m) * np.sum((pd - y) * x)
        theta_2_grad = (1 / m) * np.sum((pd - y) * (x ** 2))
        theta_3_grad = (1 / m) * np.sum((pd - y) * (x ** 3))
        theta_4_grad = (1 / m) * np.sum((pd - y) * (x ** 4))
        theta_5_grad = (1 / m) * np.sum((pd - y) * (x ** 5))

        # 更新参数
        theta[0] -= eva * theta_0_grad
        theta[1] -= eva * theta_1_grad
        theta[2] -= eva * theta_2_grad
        theta[3] -= eva * theta_3_grad
        theta[4] -= eva * theta_4_grad
        theta[5] -= eva * theta_5_grad
        cost[i] = compute_cost(x, y, theta)
    return theta, cost



eva = 1e-3  # 学习率
time = 50000  # 迭代次数

# 梯度下降
theta_gd, cost = gradient_descent(x_standardized, y, theta, eva, time)

# 预测结果
y_pd_gd = theta_gd[0] + theta_gd[1] * x_standardized + theta_gd[2] * (x_standardized ** 2) + \
            theta_gd[3] * (x_standardized ** 3) + theta_gd[4] * (x_standardized ** 4) + theta_gd[5] * (
                        x_standardized ** 5)

# 绘制拟合结果
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)', color='b', linewidth=2)
plt.plot(x, y_pd_gd, label='data', color='r', linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# 打印最终损失
print(f"在 {time} 次后的最终损失为: {cost[-1]}")
