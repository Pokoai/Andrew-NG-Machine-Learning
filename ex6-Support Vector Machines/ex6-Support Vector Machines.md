# 支持向量机
在本练习中，我们将使用高斯核函数的支持向量机（SVM）来构建垃圾邮件分类器。

[sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)

[cmap color](https://matplotlib.org/examples/color/colormaps_reference.html)

## 数据集


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
```


```python
path = '数据集/ex6data1.mat'
raw_data = loadmat(path)
```


```python
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
```


```python
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
      <th>X1</th>
      <th>X2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.9643</td>
      <td>4.5957</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.2753</td>
      <td>3.8589</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.9781</td>
      <td>4.5651</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.9320</td>
      <td>3.5519</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5772</td>
      <td>2.8560</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 绘制图像
plt.scatter(data.X1, data.X2, c=data.y)
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121704.png)



```python
# 用plot库绘制
def plot_init_pic(data, fig, ax):
    positive = data.loc[data['y']==1]
    negative = data.loc[data['y']==0]
    
    ax.scatter(positive['X1'], positive['X2'], s=50, marker='+', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
```


```python
fig, ax = plt.subplots(figsize=(12, 8))
plot_init_pic(data, fig, ax)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121712.png)


请注意，还有一个异常的正例在其他样本之外。  
这些类仍然是线性分离的，但它非常紧凑。 我们要训练线性支持向量机来学习类边界。

## try C = 1


```python
from sklearn import svm

# 配置LinearSVC参数
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=20000)
svc
```




    LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='hinge', max_iter=20000, multi_class='ovr',
              penalty='l2', random_state=None, tol=0.0001, verbose=0)




```python
# 将之前配置好的模型应用到数据集上
svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])
```




    0.9803921568627451



### 找出决策边界再绘制


```python
# 法一： 组建网格然后将网格点带入决策边界函数，找出值近似为0的点就是边界点
def find_decision_boundary(svc, x1min, x1max, x2min, x2max, diff):
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)
    
    coordinates = [(x, y) for x in x1 for y in x2]
    x_cord, y_cord = zip(*coordinates)
    c_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    c_val['val'] = svc.decision_function(c_val[['x1', 'x2']])
    
    decision = c_val[np.abs(c_val['val']) < diff]
    
    return decision.x1, decision.x2
```


```python
x1, x2 = find_decision_boundary(svc, 0, 5, 1.5, 5, 2 * 10**-3)

# fig. ax = plt.subplots(figsize=(12, 8))  逗号写成了点，jupyter发现不了
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x1, x2, s=1, c='r', label='Boundary')

plot_init_pic(data, fig, ax)

ax.set_title('SVM(C=1) Decition Boundary')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()

plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121721.png)



```python
# The confidence score for a sample is the signed distance of that sample to the hyperplane.
data['SVM1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
```


```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='seismic')
ax.set_title('SVM(C=1) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 法二：决策边界, 使用等高线表示
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=1)

plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121728.png)


## try C = 400
C对应正则化的$\lambda$，$C = \frac{1}{\lambda}$，C越大越容易过拟合。图像中最左侧的点被划分到右侧。


```python
svc1 = svm.LinearSVC(C=400, loss='hinge', max_iter=20000)
svc1.fit(data[['X1', 'X2']], data['y'])
svc1.score(data[['X1', 'X2']], data['y'])
```

    C:\Users\humin\anaconda3\lib\site-packages\sklearn\svm\_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)





    1.0



### 决策边界


```python
x1, x2 = find_decision_boundary(svc1, 0, 5, 1.5, 5, 8 * 10**-3)  # 这里调整了diff这个阈值，否则决策点连不成一条连续的线

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x1, x2, s=10, c='r', label='Boundary')

plot_init_pic(data, fig, ax)

ax.set_title('SVM(C=400) Decition Boundary')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()

plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121735.png)



```python
# The confidence score for a sample is the signed distance of that sample to the hyperplane.
data['SVM400 Confidence'] = svc1.decision_function(data[['X1', 'X2']])
```


```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM400 Confidence'], cmap='seismic')
ax.set_title('SVM(C=400) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc1.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)

plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121740.png)


# 高斯核函数 SVM with Gaussian Kernels
![](https://img.arctee.cn/202121242137-f.png)


```python
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.power(x1-x2, 2).sum() / (2 * sigma**2))
```


```python
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

gaussian_kernel(x1, x2, sigma)
```




    0.32465246735834974



## 数据集2
接下来，在另一个数据集上使用高斯内核，找非线性边界。


```python
raw_data = loadmat('数据集/ex6data2.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
```


```python
fig, ax = plt.subplots(figsize=(12, 8))
plot_init_pic(data, fig, ax)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121746.png)



```python
svc2 = svm.SVC(C=100, gamma=10, probability=True)
svc2
```




    SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=10, kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
        verbose=False)




```python
svc2.fit(data[['X1', 'X2']], data['y'])
svc2.score(data[['X1', 'X2']], data['y'])
```




    0.9698725376593279



## 直接绘制决策边界


```python
# 法二：利用等高线绘制决策边界
def plot_decision_boundary(svc, x1min, x1max, x2min, x2max, ax):
#     x1 = np.arange(x1min, x1max, 0.001)
#     x2 = np.arange(x2min, x2max, 0.001)
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)
    
    x1, x2 = np.meshgrid(x1, x2)
    y_pred = np.array([svc.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
        
    ax.contour(x1, x2, y_pred, colors='r', linewidths=5)
```


```python
fig, ax = plt.subplots(figsize=(12, 8))

plot_init_pic(data, fig, ax)
plot_decision_boundary(svc2, 0, 1, 0.4, 1, ax)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()

# 10秒
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121753.png)



```python
# 法一 要15秒
x1, x2 = find_decision_boundary(svc2, 0, 1, 0.4, 1, 0.01)  # 这里调整了diff这个阈值，否则决策点连不成一条连续的线

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x1, x2, s=5, c='r', label='Boundary')

plot_init_pic(data, fig, ax)

ax.set_title('SVM(C=400) Decition Boundary')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()

plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121758.png)


##  数据集3
对于第三个数据集，我们给出了训练和验证集，并且基于验证集性能为SVM模型找到最优超参数。  
我们现在需要寻找最优和，候选数值为[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


```python
raw_data = loadmat('数据集/ex6data3.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']

Xval = raw_data['Xval']
yval = raw_data['yval']
```


```python
fig, ax = plt.subplots(figsize=(12, 8))
plot_init_pic(data, fig, ax)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121806.png)


### 找最优超参数


```python
C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = {'C':None, 'gamma':None}
for c in C_values:
    for gamma in gamma_values:
        svc = svm.SVC(C=c, gamma=gamma, probability=True)  
        svc.fit(data[['X1', 'X2']], data['y']) # 用训练集训练
        score =  svc.score(Xval, yval)  # 用验证集选优
        if score > best_score:
            best_score = score
            best_params['C'] = c
            best_params['gamma'] = gamma
best_score, best_params
```




    (0.965, {'C': 0.3, 'gamma': 100})



## 绘制决策曲线


```python
svc3 = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], probability=True)
svc3
```




    SVC(C=0.3, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=100, kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
        verbose=False)




```python
svc3.fit(data[['X1', 'X2']], data['y'])
svc3.score(data[['X1', 'X2']], data['y'])
```




    0.95260663507109




```python
fig, ax = plt.subplots(figsize=(12, 8))

plot_init_pic(data, fig, ax)
plot_decision_boundary(svc3, -0.6, 0.3, -0.7, 0.6, ax)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.legend()
plt.show()
```


![png](https://img.arctee.cn/qiniu_picgo/20210125121813.png)


# 垃圾邮件处理
在这一部分中，我们的目标是使用SVM来构建垃圾邮件过滤器。 

特征提取的思路：

首先对垃圾邮件进行预处理：

Lower-casing

Stripping HTML

Normalizing URLs

Normalizing Email Addresses

Normalizing Numbers

Normalizing Dollars

Word Stemming

Removal of non-words

然后统计所有的垃圾邮件中单词出现的频率，提取频率超过100次的单词，得到一个单词列表。

将每个单词替换为列表中对应的编号。

提取特征：每个邮件对应一个n维向量$R^n$，$x_i \in {0, 1}$，如果第i个单词出现，则$x_i=1$，否则$x_i=0$

本文偷懒直接使用已经处理好的特征和数据...


```python
spam_train = loadmat('数据集/spamTrain.mat')
spam_test = loadmat('数据集/spamTest.mat')

spam_train
spam_train.keys(), spam_test.keys()   # 这个好，不用把所有数据打印出来就能直接看到数据的标签
```




    (dict_keys(['__header__', '__version__', '__globals__', 'X', 'y']),
     dict_keys(['__header__', '__version__', '__globals__', 'Xtest', 'ytest']))




```python
X = spam_train['X']
Xtest = spam_test['Xtest']

y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

X.shape, y.shape, Xtest.shape, ytest.shape
```




    ((4000, 1899), (4000,), (1000, 1899), (1000,))



每个文档已经转换为一个向量，其中1,899个维对应于词汇表中的1,899个单词。 它们的值为二进制，表示文档中是否存在该单词。


```python
svc4 = svm.SVC()
svc4.fit(X, y)
```




    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)




```python
print('Training accuracy = {0}%'.format(np.round(svc4.score(X, y) * 100, 2)))
print('Test accuracy = {0}%'.format(np.round(svc4.score(Xtest, ytest) * 100, 2)))
```

    Training accuracy = 99.32%
    Test accuracy = 98.7%


## 找出垃圾邮件敏感单词


```python
kw = np.eye(1899)  # 为每个单词生成一个向量，每一行代表一个单词
spam_val = pd.DataFrame({'idx':range(1899)})

print(kw[:3,:])
```

    [[1. 0. 0. ... 0. 0. 0.]
     [0. 1. 0. ... 0. 0. 0.]
     [0. 0. 1. ... 0. 0. 0.]]



```python
spam_val['isspam'] = svc4.decision_function(kw)

spam_val.head()
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
      <th>idx</th>
      <th>isspam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.093653</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.083078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-0.109401</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.119685</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.165824</td>
    </tr>
  </tbody>
</table>
</div>




```python
spam_val['isspam'].describe()  
```




    count    1899.000000
    mean       -0.110039
    std         0.049094
    min        -0.428396
    25%        -0.131213
    50%        -0.111985
    75%        -0.091973
    max         0.396286
    Name: isspam, dtype: float64




```python
decision = spam_val[spam_val['isspam'] > 0] # 提取出垃圾邮件敏感单词
decision
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
      <th>idx</th>
      <th>isspam</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>155</th>
      <td>155</td>
      <td>0.095529</td>
    </tr>
    <tr>
      <th>173</th>
      <td>173</td>
      <td>0.066666</td>
    </tr>
    <tr>
      <th>297</th>
      <td>297</td>
      <td>0.396286</td>
    </tr>
    <tr>
      <th>351</th>
      <td>351</td>
      <td>0.023785</td>
    </tr>
    <tr>
      <th>382</th>
      <td>382</td>
      <td>0.030317</td>
    </tr>
    <tr>
      <th>476</th>
      <td>476</td>
      <td>0.042474</td>
    </tr>
    <tr>
      <th>478</th>
      <td>478</td>
      <td>0.057344</td>
    </tr>
    <tr>
      <th>529</th>
      <td>529</td>
      <td>0.060692</td>
    </tr>
    <tr>
      <th>537</th>
      <td>537</td>
      <td>0.008558</td>
    </tr>
    <tr>
      <th>680</th>
      <td>680</td>
      <td>0.109643</td>
    </tr>
    <tr>
      <th>697</th>
      <td>697</td>
      <td>0.003269</td>
    </tr>
    <tr>
      <th>738</th>
      <td>738</td>
      <td>0.092561</td>
    </tr>
    <tr>
      <th>774</th>
      <td>774</td>
      <td>0.181496</td>
    </tr>
    <tr>
      <th>791</th>
      <td>791</td>
      <td>0.040396</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>1008</td>
      <td>0.012187</td>
    </tr>
    <tr>
      <th>1088</th>
      <td>1088</td>
      <td>0.132633</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>1101</td>
      <td>0.002832</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>1120</td>
      <td>0.003076</td>
    </tr>
    <tr>
      <th>1163</th>
      <td>1163</td>
      <td>0.072045</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>1178</td>
      <td>0.012122</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>1182</td>
      <td>0.015656</td>
    </tr>
    <tr>
      <th>1190</th>
      <td>1190</td>
      <td>0.232788</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>1263</td>
      <td>0.160806</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>1298</td>
      <td>0.044018</td>
    </tr>
    <tr>
      <th>1372</th>
      <td>1372</td>
      <td>0.019640</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>1397</td>
      <td>0.218337</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>1399</td>
      <td>0.018762</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>1460</td>
      <td>0.001859</td>
    </tr>
    <tr>
      <th>1467</th>
      <td>1467</td>
      <td>0.002822</td>
    </tr>
    <tr>
      <th>1519</th>
      <td>1519</td>
      <td>0.001654</td>
    </tr>
    <tr>
      <th>1661</th>
      <td>1661</td>
      <td>0.003775</td>
    </tr>
    <tr>
      <th>1721</th>
      <td>1721</td>
      <td>0.057241</td>
    </tr>
    <tr>
      <th>1740</th>
      <td>1740</td>
      <td>0.034107</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>1795</td>
      <td>0.125143</td>
    </tr>
    <tr>
      <th>1823</th>
      <td>1823</td>
      <td>0.002071</td>
    </tr>
    <tr>
      <th>1829</th>
      <td>1829</td>
      <td>0.002630</td>
    </tr>
    <tr>
      <th>1851</th>
      <td>1851</td>
      <td>0.030662</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>1892</td>
      <td>0.052786</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>1894</td>
      <td>0.101613</td>
    </tr>
  </tbody>
</table>
</div>




```python
path = '数据集/vocab.txt'
voc = pd.read_csv(path, header=None, names=['idx', 'voc'], sep='\t')
voc.head()
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
      <th>idx</th>
      <th>voc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>aa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ab</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>abil</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>abl</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>about</td>
    </tr>
  </tbody>
</table>
</div>




```python
spamvoc = voc.loc[decision['idx']]

spamvoc
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
      <th>idx</th>
      <th>voc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>155</th>
      <td>156</td>
      <td>basenumb</td>
    </tr>
    <tr>
      <th>173</th>
      <td>174</td>
      <td>below</td>
    </tr>
    <tr>
      <th>297</th>
      <td>298</td>
      <td>click</td>
    </tr>
    <tr>
      <th>351</th>
      <td>352</td>
      <td>contact</td>
    </tr>
    <tr>
      <th>382</th>
      <td>383</td>
      <td>credit</td>
    </tr>
    <tr>
      <th>476</th>
      <td>477</td>
      <td>dollar</td>
    </tr>
    <tr>
      <th>478</th>
      <td>479</td>
      <td>dollarnumb</td>
    </tr>
    <tr>
      <th>529</th>
      <td>530</td>
      <td>email</td>
    </tr>
    <tr>
      <th>537</th>
      <td>538</td>
      <td>encod</td>
    </tr>
    <tr>
      <th>680</th>
      <td>681</td>
      <td>free</td>
    </tr>
    <tr>
      <th>697</th>
      <td>698</td>
      <td>futur</td>
    </tr>
    <tr>
      <th>738</th>
      <td>739</td>
      <td>guarante</td>
    </tr>
    <tr>
      <th>774</th>
      <td>775</td>
      <td>here</td>
    </tr>
    <tr>
      <th>791</th>
      <td>792</td>
      <td>hour</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>1009</td>
      <td>market</td>
    </tr>
    <tr>
      <th>1088</th>
      <td>1089</td>
      <td>nbsp</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>1102</td>
      <td>nextpart</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>1121</td>
      <td>numbera</td>
    </tr>
    <tr>
      <th>1163</th>
      <td>1164</td>
      <td>offer</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>1179</td>
      <td>opt</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>1183</td>
      <td>order</td>
    </tr>
    <tr>
      <th>1190</th>
      <td>1191</td>
      <td>our</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>1264</td>
      <td>pleas</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>1299</td>
      <td>price</td>
    </tr>
    <tr>
      <th>1372</th>
      <td>1373</td>
      <td>receiv</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>1398</td>
      <td>remov</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>1400</td>
      <td>repli</td>
    </tr>
    <tr>
      <th>1460</th>
      <td>1461</td>
      <td>se</td>
    </tr>
    <tr>
      <th>1467</th>
      <td>1468</td>
      <td>see</td>
    </tr>
    <tr>
      <th>1519</th>
      <td>1520</td>
      <td>sincer</td>
    </tr>
    <tr>
      <th>1661</th>
      <td>1662</td>
      <td>text</td>
    </tr>
    <tr>
      <th>1721</th>
      <td>1722</td>
      <td>transfer</td>
    </tr>
    <tr>
      <th>1740</th>
      <td>1741</td>
      <td>type</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>1796</td>
      <td>visit</td>
    </tr>
    <tr>
      <th>1823</th>
      <td>1824</td>
      <td>websit</td>
    </tr>
    <tr>
      <th>1829</th>
      <td>1830</td>
      <td>welcom</td>
    </tr>
    <tr>
      <th>1851</th>
      <td>1852</td>
      <td>will</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>1893</td>
      <td>you</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>1895</td>
      <td>your</td>
    </tr>
  </tbody>
</table>



