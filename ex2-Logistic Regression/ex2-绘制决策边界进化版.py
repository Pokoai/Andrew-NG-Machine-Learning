import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['test1', 'test2', 'labels'])
# data2.head()

# positive2 = data_init[data_init['Accepted'].isin([1])]
# negative2 = data_init[data_init['Accepted'].isin([0])]
#
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive2['Test 1'], positive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
# ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()

# degree = 6
# data2 = data_init
# x1 = data2['Test 1']
# x2 = data2['Test 2']
#
# data2.insert(3, 'Ones', 1)
#
# for i in range(1, degree+1):
#     for j in range(0, i+1):
#         data2['F' + str(i-j) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
# #此处原答案错误较多，已经更正
#
# data2.drop('Test 1', axis=1, inplace=True)
# data2.drop('Test 2', axis=1, inplace=True)

# print(data2.head())

def feature_mapping(x, y, power, as_ndarray=False):
    data = {'f{0}{1}'.format(i-p, p): np.power(x, i-p) * np.power(y, p)
                for i in range(0, power+1)
                for p in range(0, i+1)
           }
    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)

# 实现sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 正则代价函数
def costReg(theta, X, y, lambdaRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = lambdaRate / (2 * len(X)) * np.sum(np.power(theta[:, 1:], 2))  # 注意是从theta1 开始惩罚的
    return np.sum(first - second) / len(X) + reg


# 实现正则化的梯度函数
def gradientReg(theta, X, y, lambdaRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    grad = (1 / len(X)) * (sigmoid(X * theta.T) - y).T * X
    reg = np.zeros(X.shape[1])
    reg[0] = 0
    reg[1:] = (lambdaRate / len(X)) * theta[:, 1:]

    return grad + reg

# 初始化X，y，θ
x1 = data2.test1.values
x2 = data2.test2.values
X2 = feature_mapping(x1, x2, power=6, as_ndarray=True)
y2 = np.array(data2.iloc[:, -1:])
theta2 = np.zeros(X2.shape[1])

# λ设为1
learningRate = 1

# 计算初始代价
# J = costReg(theta2, X2, y2, learningRate)
# print(J)

# result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
# print(result2)

# 得到theta
def find_theta(power, lambdaRate):
    '''
    power: int
        raise x1, x2 to polynomial power
    lambdaRate: int
        lambda constant for regularization term
    '''
    path = 'ex2data2.txt'
    df = pd.read_csv(path, header=None, names=['test1', 'test2', 'labels'])

    x1 = df.test1.values
    x2 = df.test2.values

    X2 = feature_mapping(x1, x2, power, as_ndarray=True)
    y2 = np.array(df.iloc[:, -1:])
    theta2 = np.zeros(X2.shape[1])

    res = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, lambdaRate))
    return res[0]


# 画决策边界
def draw_boundary(power, lambdaRate, density):

    final_theta = find_theta(power, lambdaRate)

    x = np.linspace(-1, 1.5, density)  # x坐标
    y = np.linspace(-1, 1.5, density)  # y坐标
    xx, yy = np.meshgrid(x, y)  # 生成网格数据

    # 生成高维特征数据并求出z
    z = feature_mapping(xx.ravel(), yy.ravel(), 6, as_ndarray=True) @ final_theta

    # 保持维度一致
    z = z.reshape(xx.shape)

    # 散点图
    df = pd.read_csv(path, header=None, names=['test1', 'test2', 'labels'])
    pos = df.loc[data2['labels'] == 1]
    neg = df.loc[data2['labels'] == 0]

    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = 'False'

    plt.figure(figsize=(10, 6))
    # 画等高线图
    plt.contour(xx, yy, z, 0, s=50, c='r', marker='.', label='Decision Boundary')

    plt.scatter(pos.test1, pos.test2, s=50, c='b', marker='o', label='1')
    plt.scatter(neg.test1, neg.test2, s=50, c='g', marker='x', label='0')

    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')

    end = time.perf_counter()
    runtime = round(end-start, 2)
    plt.title('用时:' + str(runtime) + ' s')

    plt.legend(loc=0)

    plt.show()

# 绘制决策边界
if __name__ == '__main__':
    start = time.perf_counter()

    draw_boundary(power=6, lambdaRate=1, density=1000)


