import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
# 设置主标题
plt.suptitle('My Figure')
plt.plot([x for x in range(10)], label='train acc')  # acc最新版keras已经无法使用
plt.plot([x for x in range(10, 20)], label='val acc')  # acc最新版keras已经无法使用
plt.title('accuracy')  # 图名
plt.ylabel('Accuracy')  # 纵坐标名
plt.xlabel('Epoch')
# 设置子标题

plt.show()
