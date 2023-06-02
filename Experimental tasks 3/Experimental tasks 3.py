import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# 读取图片
img = plt.imread('实验3图片.jpg')

# 检查图片是否成功读取
if img is not None:
    print("图片已成功读取")
else:
    print("图片读取失败，请检查图片路径和文件名是否正确")

# 显示图片
plt.imshow(img)
plt.show()

# 左右翻转图片
img1 = np.fliplr(img)

# 显示翻转后的图片
plt.imshow(img1)
plt.show()

# 缩小图片至原来的五分之一
zoom_factor = 1/5
img2 = ndimage.zoom(img, (zoom_factor, zoom_factor, 1), order=1)

# 显示缩小后的图片
plt.imshow(img2)
plt.show()

# 生成噪声
noise = np.random.randint(0, 100, img.shape, dtype=np.int32)

# 叠加噪声到图片
img3 = np.add(img.astype(np.int32), noise)

# 确保像素值在0~255之间
img3 = np.clip(img3, 0, 255).astype(np.uint8)

# 显示添加噪声后的图片
plt.imshow(img3)
plt.show()

# 生成随机颜色
random_color = np.random.randint(0, 256, size=3)

# 添加50*50像素的方块
img4 = img.copy()
img4[0:50, 0:50] = random_color

# 显示添加方块后的图片
plt.imshow(img4)
plt.show()

# 读取文件
with np.load('实验3数据.npz', allow_pickle=True) as data_file:
    data = data_file['arr_0']
    name = data_file['arr_1']

print("Data:")
print(data)
print("\nName:")
print(name)

# 删除第一列数据（年份）和倒数两行
p_data = data[:-2, 1:]

print("data:")
print(p_data)

# 计算总人口数的增长率
population_growth_rate = np.diff(p_data[::-1, 0]) / p_data[-2::-1, 0] * 100

# 将最开始年份（最大年份）的增长率设置为其下一年的增长率
population_growth_rate = np.insert(population_growth_rate, 0, population_growth_rate[0])

# 将增长率添加到p_data数组中
p_data = np.column_stack((p_data, population_growth_rate[::-1]))

print("每年总人口数的增长率为:")
print(p_data)

# 计算男女比例
gender_ratio = p_data[:, 1] / p_data[:, 2]

# 计算城乡人口比例
urban_rural_ratio = p_data[:, 3] / p_data[:, 4]

# 将男女比例和城乡人口比例添加到p_data数组中
p_data = np.column_stack((p_data, gender_ratio, urban_rural_ratio))

# 保存结果到population.npy文件
np.save('population.npy', p_data)

print("男女比例和城乡人口比例:")
print(p_data)

# 读取population.npy文件
loaded_population = np.load('population.npy', allow_pickle=True)

print("Loaded population array:")
print(loaded_population)