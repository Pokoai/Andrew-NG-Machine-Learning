> 寻找志同道合的学习伙伴，请访问我的[个人网页](https://arctee.cn/512.html).
> 该内容同步发布在[CSDN](https://blog.csdn.net/weixin_42723246)和[耳壳网](https://arctee.cn/).



# 逻辑回归（Logistic Regression）

在训练的初始阶段，我们将要构建一个逻辑回归模型来预测，某个学生是否被大学录取。  
设想你是大学相关部分的管理者，想通过申请学生两次测试的评分，来决定他们是否被录取。  
现在你拥有之前申请学生的可以用于训练逻辑回归的训练样本集。对于每一个训练样本，你有他们两次测试的评分和最后是被录取的结果。  



## UI界面展示

实现了在界面上直接更改Power和lambda值，点击’决策边界‘即可自动绘制边界，可动态调整lambda值观察过拟合、欠拟合等情况

![ex2-ui](https://img.arctee.cn/qiniu_picgo/20210117073403.jpg)

> UI完整代码放在[Github](https://github.com/adairhu/Andrew-NG-Meachine-Learning/tree/main/ex2-Logistic%20Regression)上，欢迎Star收藏~



## 数据可视化


```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# 读取文件
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exam1</th>
      <th>exam2</th>
      <th>admitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.623660</td>
      <td>78.024693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.286711</td>
      <td>43.894998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.847409</td>
      <td>72.902198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.182599</td>
      <td>86.308552</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79.032736</td>
      <td>75.344376</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exam1</th>
      <th>exam2</th>
      <th>admitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>65.644274</td>
      <td>66.221998</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>19.458222</td>
      <td>18.582783</td>
      <td>0.492366</td>
    </tr>
    <tr>
      <th>min</th>
      <td>30.058822</td>
      <td>30.603263</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>50.919511</td>
      <td>48.179205</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>67.032988</td>
      <td>67.682381</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80.212529</td>
      <td>79.360605</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>99.827858</td>
      <td>98.869436</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 按类别分割数据集
positive = data.loc[data['admitted']==1]
negative = data.loc[data['admitted']==0]

negative.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exam1</th>
      <th>exam2</th>
      <th>admitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.623660</td>
      <td>78.024693</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.286711</td>
      <td>43.894998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.847409</td>
      <td>72.902198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>45.083277</td>
      <td>56.316372</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>95.861555</td>
      <td>38.225278</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
positive.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>exam1</th>
      <th>exam2</th>
      <th>admitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>60.182599</td>
      <td>86.308552</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>79.032736</td>
      <td>75.344376</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>61.106665</td>
      <td>96.511426</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>75.024746</td>
      <td>46.554014</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>76.098787</td>
      <td>87.420570</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 绘制图像
plt.figure(figsize=(10, 6))
plt.scatter(positive.exam1, positive.exam2, c='b', marker='o', label='Admitted')
plt.scatter(negative.exam1, negative.exam2, c='r', marker='x', label='Not Admitted')
plt.xlabel('Exam1 Score')
plt.ylabel('Exam2 Score')
plt.legend(loc=1)
plt.show()
```

![output_7_0](https://img.arctee.cn/qiniu_picgo/20210117072658.png)

```python
# 方式二：直接用data值进行绘制，不用将数据集分开，但是此时只能区分颜色，无法区分形状
plt.figure(figsize=(10, 6))
plt.scatter(data.exam1, data.exam2, c=data.admitted)
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210117072710.png)



```python
# 因为两个特征的量级一样且数值相差不大，故不用归一化
```

## Sigmoid函数
![image.png](attachment:image.png)


```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```


```python
x = np.arange(-10, 10, 0.5)

plt.plot(x, sigmoid(x), 'r')
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210117072715.png)


## 代价函数
![image.png](attachment:image.png)


```python
# 实现代价函数
def costFunction(theta, X, y):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)
    
    first = np.multiply(- y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / len(X) 
```


```python
a = np.mat('1 2 3 4')
b = np.mat('2 2 2 2')

np.multiply(a, b)
```




    matrix([[2, 4, 6, 8]])



## 变量初始化


```python
# 加一列
data.insert(0, 'Ones', 1)

# 初始化变量
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]

# 转化为数组array类型
X = np.array(X)
y = np.array(y)
theta = np.zeros(X.shape[1])
```


```python
# 检查矩阵维数
X.shape, y.shape, theta.shape
```




    ((100, 3), (100, 1), (3,))




```python
costFunction(theta, X, y)
```




    0.6931471805599453



## 梯度函数
![image.png](attachment:image.png)


```python
## 实现梯度函数(并没有更新θ)
def gradient(theta, X, y):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)
    
    m = len(X)
    dtheta = (sigmoid(X * theta.T) - y).T * X
    return dtheta / m
```


```python
gradient(theta, X, y)
```




    matrix([[ -0.1       , -12.00921659, -11.26284221]])



## 用工具箱计算θ的值
在此前的线性回归中，我们自己写代码实现的梯度下降（ex1的2.2.4的部分）。当时我们写了一个代价函数、计算了他的梯度，然后对他执行了梯度下降的步骤。这次，我们不执行梯度下降步骤，而是从scipy库中调用了一个内置的函数fmin_tnc。这就是说，我们不用自己定义迭代次数和步长，功能会直接告诉我们最优解。            

andrew ng在课程中用的是Octave的“fminunc”函数，由于我们使用Python，我们可以用scipy.optimize.fmin_tnc做同样的事情。
（另外，如果对fminunc有疑问的，可以参考下面这篇百度文库的内容https://wenku.baidu.com/view/2f6ce65d0b1c59eef8c7b47a.html ）
如果一切顺利的话，最有θ对应的代价应该是0.203


```python
import scipy.optimize as opt
result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X, y))
result
```




    (array([-25.16131849,   0.20623159,   0.20147149]), 36, 0)




```python
result[0][0],result[0][1],result[0][2]
```




    (-25.161318491498374, 0.206231587441317, 0.2014714850703228)



## 决策边界
![image.png](attachment:image.png)


```python
x1 = np.linspace(30, 100, 100)
y_ = -(result[0][0] + result[0][1] * x1) / result[0][2]

plt.figure(figsize=(10, 6))
plt.plot(x1, y_)
plt.scatter(positive.exam1, positive.exam2, c='b', marker='o', label='Admitted')
plt.scatter(negative.exam1, negative.exam2, c='r', marker='x', label='Not Admitted')
plt.xlabel('Exam1 Score')
plt.ylabel('Exam2 Score')
plt.legend(loc=1)
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210117072726.png)


## 评价逻辑回归模型
写一个predict的函数，给出数据以及参数后，会返回“1”或者“0”。然后再把这个predict函数用于训练集上，看准确率怎样。  

![image.png](attachment:image.png)



```python
# 预测函数
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
```


```python
# 模型正确率
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if a^b == 0 else 0 for (a,b) in zip(predictions, y)]
accuracy = (sum(correct) / len(correct))
print('accuracy = {0:.0f}%'.format(accuracy*100))
```

    accuracy = 89%


# 正则化逻辑回归

## 数据可视化


```python
path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['test1', 'test2', 'labels'])
data2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test1</th>
      <th>test2</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.051267</td>
      <td>0.69956</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.092742</td>
      <td>0.68494</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.213710</td>
      <td>0.69225</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.375000</td>
      <td>0.50219</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.513250</td>
      <td>0.46564</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pos = data2.loc[data2['labels']==1]
neg = data2.loc[data2['labels']==0]

neg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>test1</th>
      <th>test2</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>0.18376</td>
      <td>0.93348</td>
      <td>0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.22408</td>
      <td>0.77997</td>
      <td>0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.29896</td>
      <td>0.61915</td>
      <td>0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.50634</td>
      <td>0.75804</td>
      <td>0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.61578</td>
      <td>0.72880</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 6))
plt.scatter(pos.test1, pos.test2, c='b', marker='o', label='1')
plt.scatter(neg.test1, neg.test2, c='r', marker='x', label='0')

plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend(loc=0)

plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210117072734.png)


以上图片显示，这个数据集不能像之前一样使用直线将两部分分割。而逻辑回归只适用于线性的分割，所以，这个数据集不适合直接使用逻辑回归。

## 特征映射
一种更好的使用数据集的方式是为每组数据创造更多的特征。所以我们为每组添加了最高到6次幂的特征
![image.png](attachment:image.png)


```python
def feature_mapping(x, y, power, as_ndarray=False):
    data = {'f{0}{1}'.format(i-p, p): np.power(x, i-p) * np.power(y, p)
                for i in range(0, power+1)
                for p in range(0, i+1)
           }
    if as_ndarray:
        return pd.DataFrame(data).values
    else:
        return pd.DataFrame(data)
```


```python
x1 = data2.test1.values
x2 = data2.test2.values

data3 = feature_mapping(x1, x2, power=6)
data3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f00</th>
      <th>f10</th>
      <th>f01</th>
      <th>f20</th>
      <th>f11</th>
      <th>f02</th>
      <th>f30</th>
      <th>f21</th>
      <th>f12</th>
      <th>f03</th>
      <th>...</th>
      <th>f23</th>
      <th>f14</th>
      <th>f05</th>
      <th>f60</th>
      <th>f51</th>
      <th>f42</th>
      <th>f33</th>
      <th>f24</th>
      <th>f15</th>
      <th>f06</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.051267</td>
      <td>0.69956</td>
      <td>0.002628</td>
      <td>0.035864</td>
      <td>0.489384</td>
      <td>0.000135</td>
      <td>0.001839</td>
      <td>0.025089</td>
      <td>0.342354</td>
      <td>...</td>
      <td>0.000900</td>
      <td>0.012278</td>
      <td>0.167542</td>
      <td>1.815630e-08</td>
      <td>2.477505e-07</td>
      <td>0.000003</td>
      <td>0.000046</td>
      <td>0.000629</td>
      <td>0.008589</td>
      <td>0.117206</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.092742</td>
      <td>0.68494</td>
      <td>0.008601</td>
      <td>-0.063523</td>
      <td>0.469143</td>
      <td>-0.000798</td>
      <td>0.005891</td>
      <td>-0.043509</td>
      <td>0.321335</td>
      <td>...</td>
      <td>0.002764</td>
      <td>-0.020412</td>
      <td>0.150752</td>
      <td>6.362953e-07</td>
      <td>-4.699318e-06</td>
      <td>0.000035</td>
      <td>-0.000256</td>
      <td>0.001893</td>
      <td>-0.013981</td>
      <td>0.103256</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-0.213710</td>
      <td>0.69225</td>
      <td>0.045672</td>
      <td>-0.147941</td>
      <td>0.479210</td>
      <td>-0.009761</td>
      <td>0.031616</td>
      <td>-0.102412</td>
      <td>0.331733</td>
      <td>...</td>
      <td>0.015151</td>
      <td>-0.049077</td>
      <td>0.158970</td>
      <td>9.526844e-05</td>
      <td>-3.085938e-04</td>
      <td>0.001000</td>
      <td>-0.003238</td>
      <td>0.010488</td>
      <td>-0.033973</td>
      <td>0.110047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.375000</td>
      <td>0.50219</td>
      <td>0.140625</td>
      <td>-0.188321</td>
      <td>0.252195</td>
      <td>-0.052734</td>
      <td>0.070620</td>
      <td>-0.094573</td>
      <td>0.126650</td>
      <td>...</td>
      <td>0.017810</td>
      <td>-0.023851</td>
      <td>0.031940</td>
      <td>2.780914e-03</td>
      <td>-3.724126e-03</td>
      <td>0.004987</td>
      <td>-0.006679</td>
      <td>0.008944</td>
      <td>-0.011978</td>
      <td>0.016040</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>-0.513250</td>
      <td>0.46564</td>
      <td>0.263426</td>
      <td>-0.238990</td>
      <td>0.216821</td>
      <td>-0.135203</td>
      <td>0.122661</td>
      <td>-0.111283</td>
      <td>0.100960</td>
      <td>...</td>
      <td>0.026596</td>
      <td>-0.024128</td>
      <td>0.021890</td>
      <td>1.827990e-02</td>
      <td>-1.658422e-02</td>
      <td>0.015046</td>
      <td>-0.013650</td>
      <td>0.012384</td>
      <td>-0.011235</td>
      <td>0.010193</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
data3.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f00</th>
      <th>f10</th>
      <th>f01</th>
      <th>f20</th>
      <th>f11</th>
      <th>f02</th>
      <th>f30</th>
      <th>f21</th>
      <th>f12</th>
      <th>f03</th>
      <th>...</th>
      <th>f23</th>
      <th>f14</th>
      <th>f05</th>
      <th>f60</th>
      <th>f51</th>
      <th>f42</th>
      <th>f33</th>
      <th>f24</th>
      <th>f15</th>
      <th>f06</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>118.0</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>1.180000e+02</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>118.000000</td>
      <td>...</td>
      <td>118.000000</td>
      <td>1.180000e+02</td>
      <td>118.000000</td>
      <td>1.180000e+02</td>
      <td>118.000000</td>
      <td>1.180000e+02</td>
      <td>118.000000</td>
      <td>1.180000e+02</td>
      <td>118.000000</td>
      <td>1.180000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.0</td>
      <td>0.054779</td>
      <td>0.183102</td>
      <td>0.247575</td>
      <td>-0.025472</td>
      <td>0.301370</td>
      <td>5.983333e-02</td>
      <td>0.030682</td>
      <td>0.015483</td>
      <td>0.142350</td>
      <td>...</td>
      <td>0.018278</td>
      <td>4.089084e-03</td>
      <td>0.115710</td>
      <td>7.837118e-02</td>
      <td>-0.000703</td>
      <td>1.893340e-02</td>
      <td>-0.001705</td>
      <td>2.259170e-02</td>
      <td>-0.006302</td>
      <td>1.257256e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>0.496654</td>
      <td>0.519743</td>
      <td>0.248532</td>
      <td>0.224075</td>
      <td>0.284536</td>
      <td>2.746459e-01</td>
      <td>0.134706</td>
      <td>0.150143</td>
      <td>0.326134</td>
      <td>...</td>
      <td>0.058513</td>
      <td>9.993907e-02</td>
      <td>0.299092</td>
      <td>1.938621e-01</td>
      <td>0.058271</td>
      <td>3.430092e-02</td>
      <td>0.037443</td>
      <td>4.346935e-02</td>
      <td>0.090621</td>
      <td>2.964416e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0</td>
      <td>-0.830070</td>
      <td>-0.769740</td>
      <td>0.000040</td>
      <td>-0.484096</td>
      <td>0.000026</td>
      <td>-5.719317e-01</td>
      <td>-0.358121</td>
      <td>-0.483743</td>
      <td>-0.456071</td>
      <td>...</td>
      <td>-0.142660</td>
      <td>-4.830370e-01</td>
      <td>-0.270222</td>
      <td>6.472253e-14</td>
      <td>-0.203971</td>
      <td>2.577297e-10</td>
      <td>-0.113448</td>
      <td>2.418097e-10</td>
      <td>-0.482684</td>
      <td>1.795116e-14</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.0</td>
      <td>-0.372120</td>
      <td>-0.254385</td>
      <td>0.043243</td>
      <td>-0.178209</td>
      <td>0.061086</td>
      <td>-5.155632e-02</td>
      <td>-0.023672</td>
      <td>-0.042980</td>
      <td>-0.016492</td>
      <td>...</td>
      <td>-0.001400</td>
      <td>-7.449462e-03</td>
      <td>-0.001072</td>
      <td>8.086369e-05</td>
      <td>-0.006381</td>
      <td>1.258285e-04</td>
      <td>-0.005749</td>
      <td>3.528590e-04</td>
      <td>-0.016662</td>
      <td>2.298277e-04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0</td>
      <td>-0.006336</td>
      <td>0.213455</td>
      <td>0.165397</td>
      <td>-0.016521</td>
      <td>0.252195</td>
      <td>-2.544062e-07</td>
      <td>0.006603</td>
      <td>-0.000039</td>
      <td>0.009734</td>
      <td>...</td>
      <td>0.001026</td>
      <td>-8.972096e-09</td>
      <td>0.000444</td>
      <td>4.527344e-03</td>
      <td>-0.000004</td>
      <td>3.387050e-03</td>
      <td>-0.000005</td>
      <td>3.921378e-03</td>
      <td>-0.000020</td>
      <td>1.604015e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0</td>
      <td>0.478970</td>
      <td>0.646562</td>
      <td>0.389925</td>
      <td>0.100795</td>
      <td>0.464189</td>
      <td>1.099616e-01</td>
      <td>0.086392</td>
      <td>0.079510</td>
      <td>0.270310</td>
      <td>...</td>
      <td>0.021148</td>
      <td>2.751341e-02</td>
      <td>0.113020</td>
      <td>5.932959e-02</td>
      <td>0.002104</td>
      <td>2.090875e-02</td>
      <td>0.001024</td>
      <td>2.103622e-02</td>
      <td>0.001289</td>
      <td>1.001215e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0</td>
      <td>1.070900</td>
      <td>1.108900</td>
      <td>1.146827</td>
      <td>0.568307</td>
      <td>1.229659</td>
      <td>1.228137e+00</td>
      <td>0.449251</td>
      <td>0.505577</td>
      <td>1.363569</td>
      <td>...</td>
      <td>0.287323</td>
      <td>4.012965e-01</td>
      <td>1.676725</td>
      <td>1.508320e+00</td>
      <td>0.250577</td>
      <td>2.018260e-01</td>
      <td>0.183548</td>
      <td>2.556084e-01</td>
      <td>0.436209</td>
      <td>1.859321e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 28 columns</p>
</div>



## 正则化代价函数
![image.png](attachment:image.png)


```python
# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # 负号不要丢了
```


```python
x = np.arange(-10, 10, 0.02)
plt.plot(x, sigmoid(x))
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210117072743.png)



```python
# 正则代价函数
def costReg(theta, X, y, lambdaRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = lambdaRate / (2*len(X)) * np.sum(np.power(theta[:, 1:], 2)) # 注意是从theta1 开始惩罚的
    return np.sum(first - second) / len(X) + reg
```


```python
X2 = feature_mapping(x1, x2, power=6, as_ndarray=True)
y2 = np.array(data2.iloc[:, -1:])
theta2 = np.zeros(X2.shape[1])
```


```python
X2.shape, y2.shape, theta2.shape
```




    ((118, 28), (118, 1), (28,))




```python
X2
```




    array([[ 1.00000000e+00,  5.12670000e-02,  6.99560000e-01, ...,
             6.29470940e-04,  8.58939846e-03,  1.17205992e-01],
           [ 1.00000000e+00, -9.27420000e-02,  6.84940000e-01, ...,
             1.89305413e-03, -1.39810280e-02,  1.03255971e-01],
           [ 1.00000000e+00, -2.13710000e-01,  6.92250000e-01, ...,
             1.04882142e-02, -3.39734512e-02,  1.10046893e-01],
           ...,
           [ 1.00000000e+00, -4.84450000e-01,  9.99270000e-01, ...,
             2.34007252e-01, -4.82684337e-01,  9.95627986e-01],
           [ 1.00000000e+00, -6.33640000e-03,  9.99270000e-01, ...,
             4.00328554e-05, -6.31330588e-03,  9.95627986e-01],
           [ 1.00000000e+00,  6.32650000e-01, -3.06120000e-02, ...,
             3.51474517e-07, -1.70067777e-08,  8.22905998e-10]])




```python
costReg(theta2, X2, y2, 1)
```




    0.6931471805599454



## 正则化梯度函数
![image.png](attachment:image.png)


```python
def gradientReg(theta, X, y, lambdaRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    
    grad = (1 / len(X)) * (sigmoid(X * theta.T) - y).T * X
    reg = np.zeros(X.shape[1])
    reg[0] = 0
    reg[1:] = (lambdaRate / len(X)) * theta[:, 1:]
    
    return grad + reg
```


```python
lambdaRate = 1
gradientReg(theta2, X2, y2, lambdaRate)
```




    matrix([[8.47457627e-03, 1.87880932e-02, 7.77711864e-05, 5.03446395e-02,
             1.15013308e-02, 3.76648474e-02, 1.83559872e-02, 7.32393391e-03,
             8.19244468e-03, 2.34764889e-02, 3.93486234e-02, 2.23923907e-03,
             1.28600503e-02, 3.09593720e-03, 3.93028171e-02, 1.99707467e-02,
             4.32983232e-03, 3.38643902e-03, 5.83822078e-03, 4.47629067e-03,
             3.10079849e-02, 3.10312442e-02, 1.09740238e-03, 6.31570797e-03,
             4.08503006e-04, 7.26504316e-03, 1.37646175e-03, 3.87936363e-02]])



## 用工具库求解参数


```python
import scipy.optimize as opt

result2 = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, lambdaRate))
result2
```




    (array([ 1.27271027,  0.62529965,  1.18111686, -2.01987399, -0.91743189,
            -1.43166929,  0.12393227, -0.36553118, -0.35725403, -0.17516291,
            -1.4581701 , -0.05098418, -0.61558553, -0.27469165, -1.19271298,
            -0.2421784 , -0.20603298, -0.04466178, -0.27778951, -0.29539513,
            -0.45645981, -1.04319155,  0.02779373, -0.2924487 ,  0.0155576 ,
            -0.32742406, -0.1438915 , -0.92467487]),
     32,
     1)



## 评价逻辑函数


```python
# 预测函数
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
```


```python
# 模型正确率
theta_min = np.mat(result2[0])
predictions = predict(theta_min, X2)
correct =  [1 if((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
# accuracy = sum(map(int, correct)) % len(correct)
accuracy = sum(correct) / len(correct)
print('accuracy = {0:.0f}%'.format(accuracy*100))
```

    accuracy = 83%


## 画出决策的曲线


```python
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
```


```python
find_theta(power=6, lambdaRate=1)
```




    array([ 1.27271027,  0.62529965,  1.18111686, -2.01987399, -0.91743189,
           -1.43166929,  0.12393227, -0.36553118, -0.35725403, -0.17516291,
           -1.4581701 , -0.05098418, -0.61558553, -0.27469165, -1.19271298,
           -0.2421784 , -0.20603298, -0.04466178, -0.27778951, -0.29539513,
           -0.45645981, -1.04319155,  0.02779373, -0.2924487 ,  0.0155576 ,
           -0.32742406, -0.1438915 , -0.92467487])




```python
# 找决策边界，thetaX = 0, thetaX <= threshhold
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power, as_ndarray=False)

    pred = mapped_cord.values @ theta.T  # array数组进行矩阵乘法
    decision = mapped_cord[np.abs(pred) <= threshhold]

    return decision.f10, decision.f01 # 从28个特征中选出两个特征x1, x2
```


```python
# 画决策边界
def draw_boundary(power, lambdaRate):
    density = 2000
    threshhold = 2 * 10**(-3)
    
    theta = find_theta(power, lambdaRate)
    x, y = find_decision_boundary(density, power, theta, threshhold) 
    
    df = pd.read_csv(path, header=None, names=['test1', 'test2', 'labels'])
    pos = df.loc[data2['labels']==1]
    neg = df.loc[data2['labels']==0]
    
    plt.rcParams['font.family'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = 'False'

    plt.figure(figsize=(12, 8))
    plt.scatter(pos.test1, pos.test2, s=50, c='b', marker='o', label='1')
    plt.scatter(neg.test1, neg.test2, s=50, c='g', marker='x', label='0')
    plt.scatter(x, y, s=50, c='r', marker='.', label='Decision Boundary')

    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')
    
    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')

    end = time.perf_counter()
    runtime = round(end-start, 2)
    plt.title('用时:' + str(runtime) + ' s')
    
    plt.legend(loc=0)

    plt.show()
```


```python
start = time.perf_counter()
draw_boundary(power=6, lambdaRate=1)
```


![png](https://img.arctee.cn/qiniu_picgo/20210117072755.png)


## 改变λ，观察决策曲线

### 过拟合  λ=0


```python
draw_boundary(power=6, lambdaRate=0)
```


![png](https://img.arctee.cn/qiniu_picgo/20210117072758.png)


### 欠拟合  λ=100


```python
draw_boundary(power=6, lambdaRate=100)
```


![png](https://img.arctee.cn/qiniu_picgo/20210117074743.png)



> 寻找志同道合的学习伙伴，请访问我的[个人网页](https://arctee.cn/512.html).
> 该内容同步发布在[CSDN](https://blog.csdn.net/weixin_42723246)和[耳壳网](https://arctee.cn/).