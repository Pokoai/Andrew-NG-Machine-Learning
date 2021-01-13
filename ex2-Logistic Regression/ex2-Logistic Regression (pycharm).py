import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])

## 绘制图像
# 按类别分割数据集
# positive = data.loc[data['admitted']==1]
# negative = data.loc[data['admitted']==0]

# plt.figure(figsize=(10, 6))
# plt.scatter(positive.exam1, positive.exam2, c='b', marker='o', label='Admitted')
# plt.scatter(negative.exam1, negative.exam2, c='r', marker='x', label='Not Admitted')
# plt.xlabel('Exam1 Score')
# plt.ylabel('Exam2 Score')
# plt.legend(loc=1)
# plt.show()

## 变量初始化
# 加一列
data.insert(0, 'Ones', 1)

# 初始化变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]

'''
错误2：
！！！SciPy的sfmin_tnc不能很好地使用行向量或者列向量，它期望的参数是 数组 形式的。
'''
# X = np.mat(X)
# y = np.mat(y)
# theta = np.mat(np.zeros(3))

# 转化为数组array类型，而不是matrix类型
X = np.array(X)
y = np.array(y)
theta = np.zeros(X.shape[1])
# print(X.shape, y.shape, theta.shape)

# 实现sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 实现代价函数
'''
错误3：
！！！以后注意：一定要把参数值theta放前面，X、y是确定值放在后面，否则无法使用scipy的优化函数
找了几个小时才发现问题出在这
'''
def costFunction(theta, X, y):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)

    first = np.multiply(- y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / len(X)

# 检查代价函数是否正确
# print(costFunction(theta, X, y))

## 实现梯度函数(并没有更新θ)
def gradient(theta, X, y):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)

    m = len(X)
    dtheta = (sigmoid(X * theta.T) - y).T * X
    return dtheta / m

# 检查梯度函数
# print(gradient(theta, X, y))

# 使用scipy的优化函数 scipy.optimize.fmin_tnc
import scipy.optimize as opt
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X, y))
print(result)


