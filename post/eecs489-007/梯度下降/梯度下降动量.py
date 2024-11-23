import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 定义函数和导数
def f(x):
    return np.sin(x)

def df(x):
    return np.cos(x)

# 梯度下降初始化
x0 = 2
alpha = 0.5  # 学习率
v = 0 # 初始速度
beta = 0.9
trajectory = [x0]  # 用于存储每步的 x0

# 计算梯度下降路径
for _ in range(100):  # 限定50步
    v = beta * v + (1 - beta) * df(x0)
    x0 = x0 - alpha * v
    trajectory.append(x0)

# 创建绘图框架
fig, ax = plt.subplots()
x = np.linspace(-10, 10, 500)  # 定义函数曲线范围
y = f(x)
ax.plot(x, y, 'blue', label='f(x) = sin(x)')
ln, = ax.plot([], [], 'ro')  # 动态点
text = ax.text(-9, 0.8, '', fontsize=12)  # 动态显示当前值

def init():
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)
    ln.set_data([], [])
    text.set_text('')
    return ln, text

def update(frame):
    # 更新点的位置
    x_curr = trajectory[frame]
    ln.set_data([x_curr], [f(x_curr)])
    text.set_text(f'Step {frame}: x={x_curr:.2f}')
    return ln, text

# 创建动画
anim = animation.FuncAnimation(
    fig, update, frames=len(trajectory), interval=300, init_func=init, blit=True
)
plt.show()
# 保存动画为 GIF
anim.save('D:/Blogs/blogs/content/post/eecs489-007/梯度下降/梯度下架动量.gif',writer='pillow')
print("动画已保存为 gradient_descent.gif")
