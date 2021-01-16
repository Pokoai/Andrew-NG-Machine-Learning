import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy import optimize as opt
from sklearn.metrics import classification_report


# 读取文件
def load_mat(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']

    return X, y


# 标签重编码
def expend_y(y):
    result = []
    # 将y的每个元素转换为一个向量，标签值的位置为1，其余为0
    for label in y:
        y_array = np.zeros(10)
        y_array[label-1] = 1
        result.append(y_array)
    return np.array(result)


# 展开/合并参数
def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))

def deserialize(seq):
    return seq[:25*401].reshape(25, 401), seq[25*401:].reshape(10, 26)


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# S函数梯度
def sigmoid_gradient(z):
    return sigmoid(z) * (1- sigmoid(z))


# 前向传播
def forward_propagation(theta, X):  # theta,X都是array数组类型，非矩阵
    t1, t2 = deserialize(theta)

    a1 = X  # 5000*401

    z2 = a1 @ t1.T  # 5000*25
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, np.ones(len(a2)), axis=1)  # 5000*26

    z3 = a2 @ t2.T  # 5000*10
    a3 = sigmoid(z3)
    h = a3  # 5000*10

    return a1, z2, a2, z3, h


# 代价函数
def nnCost(theta, X, y): # 均为数组
    h = forward_propagation(theta, X)[-1]
    temp = -y * np.log(h) - (1 - y) * np.log(1- h)
    return  temp.sum() / len(X)


# 正则化代价函数
def nnCostReg(theta, X, y, lambdaRate=1):
    theta1, theta2 = deserialize(theta)

    first = nnCost(theta, X, y)

    reg1 = (np.power(theta1[:, 1:], 2)).sum()
    reg2 = (np.power(theta2[:, 1:], 2)).sum()
    reg = lambdaRate / (2 * len(X)) * (reg1 + reg2)

    return first + reg


# 反向传播-更新梯度
def nnGradient(theta, X, y):
    theta1, theta2 = deserialize(theta)
    # 首先正向传播
    a1, z2, a2, z3, h = forward_propagation(theta, X)
    # 针对每层的各个节点计算误差项
    d3 = h - y  # 输出层误差 5000*10
    d2 = d3 @ theta2[:, 1:] * sigmoid_gradient(z2)  # 隐藏层误差 5000*25

    delta1 = d2.T @ a1  # 25 * 401
    delta2 = d3.T @ a2  # 10 * 26
    delta = serialize(delta1, delta2)  # (10285, )

    return delta / len(X)  # (10285, )

# 正则化梯度函数
def nnGradientReg(theta, X, y, lambdaRate=1):
    first = nnGradient(theta, X, y)

    t1, t2 = deserialize(theta)
    t1[:, 0] = 0
    t2[:, 0] = 0

    reg = lambdaRate / len(X) * serialize(t1, t2)

    return first + reg

# 梯度检验
def gradient_checking(theta, X, y, eps, regularized=False):
    m = len(theta)

    def a_numeric_grad(plus, minus, X, y):
        if regularized:
            return (nnCostReg(plus, X, y) - nnCostReg(minus, X, y)) / (2 * eps)
        else:
            return (nnCost(plus, X, y) - nnCost(minus, X, y)) / (2 * eps)

    approxGrad = np.zeros(m)
    # 计算偏导数
    for i in range(m):
        thetaPlus = theta.copy()
        thetaPlus[i] = theta[i] + eps

        thetaMinus = theta.copy()
        thetaMinus[i] = theta[i] - eps

        grad = a_numeric_grad(thetaPlus, thetaMinus, X, y)
        approxGrad[i] = grad

    # 用梯度公式
    funcGrad = nnGradientReg(theta, X, y) if regularized else nnGradient(theta, X, y)

    diff = np.linalg.norm(approxGrad - funcGrad) / np.linalg.norm(approxGrad + funcGrad)

    print('If your backpropagation implementation is correct,\n' \
          + 'the relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\n'
          + 'Relative Difference: {}\n'.format(diff))


# 从服从均匀分布的范围中随机返回size大小的值
def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


# 训练
def nnTraining(X, y):
    # 初始化theta
    init_theta = random_init(10285)

    res = opt.minimize(fun=nnCostReg,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=nnGradientReg,
                       options={'maxiter': 300})
    return res


# 预测函数
def predict(theta, X):
    h = forward_propagation(theta, X)[-1]
    y_predict = np.argmax(h, axis=1) + 1
    return y_predict.reshape(5000, 1)


# 可视化隐层
def plot_hidden(theta):
    theta1, theta2 = deserialize(theta)
    hidden_layer = theta1[:, 1:]

    fig, ax = plt.subplots(5, 5, sharey=True, sharex=True, figsize=(4, 3))
    for r in range(5):
        for c in range(5):
            ax[r, c].matshow(np.array(hidden_layer[5 * r + c].reshape(20, 20)).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

if __name__ == '__main__':
    # 获取训练数据集
    raw_X, raw_y = load_mat('ex4data1.mat')

    # 处理X, y
    X = np.insert(raw_X, 0, np.ones(len(raw_X)), axis=1)
    y = expend_y(raw_y)

    # 开始训练
    res = nnTraining(X, y)
    final_theta = res.x

    # 预测
    y_predict= predict(final_theta, X)

    # 打印报告
    # print(classification_report(raw_y, y_predict))

    # 隐层可视化
    plot_hidden(final_theta)

