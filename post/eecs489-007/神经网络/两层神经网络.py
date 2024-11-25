import numpy as np
 
# N是训练的batch size; D_in 是input输入数据的维度;
# H是隐藏层的节点数; D_out 输出的维度，即输出节点数.
N, D_in, H, D_out = 64, 1000, 100, 10
 
# 创建输入、输出数据
x = np.random.randn(N, D_in)  #（64，1000）
y = np.random.randn(N, D_out) #（64，10）可以看成是一个10分类问题
 
# 权值初始化
w1 = np.random.randn(D_in, H)  #(1000,100),即输入层到隐藏层的权重
w2 = np.random.randn(H, D_out) #(100,10),即隐藏层到输出层的权重
 
learning_rate = 1e-6   #学习率
 
for t in range(500):
    # 第一步：数据的前向传播，计算预测值p_pred
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
 
    # 第二步：计算计算预测值p_pred与真实值的误差
    loss = np.square(y_pred - y).sum()
    print(t, loss)
 
    # 第三步：反向传播误差，更新两个权值矩阵
    grad_y_pred = 2.0 * (y_pred - y)     #注意：这里的导函数也是自己求的，因为这个地方是很简答的函数表达式
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
 
    # 更新参数
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
