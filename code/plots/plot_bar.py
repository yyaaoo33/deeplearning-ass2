import matplotlib.pyplot as plt
import numpy as np

models = ['ResNet18', 'EfficientNet', 'MobileNet', 'VGG16']

pretrain_accuracies = [0.82, 0.8053, 0.7880, 0.8640]
without_pretrain_accuracies = [0.577, 0.4693, 0.3732, 0.7003]

bar_width = 0.35
index = np.arange(len(models))

fig, ax = plt.subplots()

bars1 = ax.bar(index, pretrain_accuracies, bar_width, label='Pretrained', color='b')
bars2 = ax.bar(index + bar_width, without_pretrain_accuracies, bar_width, label='Without Pretrained', color='r')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy with and without Pretraining')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.savefig('bar.pdf', format='pdf', dpi=300)

plt.show()