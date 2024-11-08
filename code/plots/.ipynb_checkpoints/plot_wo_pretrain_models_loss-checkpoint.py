import matplotlib.pyplot as plt

epochs = list(range(1, 21))

resnet18_train_loss = [
    1.5926, 1.1967, 0.9666, 0.7543, 0.5408, 0.3608, 0.2306,
    0.1680, 0.1399, 0.1162, 0.1110, 0.1118, 0.0968, 0.0938,
    0.0836, 0.0851, 0.0733, 0.0660, 0.0923, 0.0686
]
resnet18_valid_loss = [
    1.4048, 1.3109, 1.2626, 1.3804, 1.4176, 1.6172, 1.7560,
    1.8781, 1.8881, 1.9272, 2.1454, 2.0588, 2.1251, 2.1772,
    2.1625, 2.1711, 2.1434, 2.4379, 2.1903, 2.3232
]

efficientnet_b0_train_loss = [
    2.2509, 2.0945, 1.9710, 1.8614, 1.7704, 1.6915, 1.6213,
    1.5636, 1.5019, 1.4418, 1.3905, 1.3360, 1.2788, 1.2269,
    1.1778, 1.1304, 1.0809, 1.0241, 0.9820, 0.9321
]
efficientnet_b0_valid_loss = [
    2.1573, 2.0126, 1.9105, 1.8147, 1.7295, 1.6817, 1.6417,
    1.6061, 1.5656, 1.5481, 1.5347, 1.5278, 1.5186, 1.5362,
    1.5434, 1.5623, 1.5700, 1.6037, 1.6159, 1.6439
]

mobilenet_train_loss = [
    2.2321, 2.0985, 2.0284, 1.9724, 1.9039, 1.8495, 1.7861,
    1.7322, 1.6807, 1.6257, 1.5810, 1.5316, 1.4852, 1.4396,
    1.3956, 1.3535, 1.3102, 1.2719, 1.2260, 1.1831
]
mobilenet_valid_loss = [
    2.1358, 2.0566, 2.0143, 1.9611, 1.9134, 1.8804, 1.8417,
    1.8154, 1.7895, 1.7921, 1.7730, 1.7639, 1.7614, 1.7475,
    1.7636, 1.7777, 1.7787, 1.7978, 1.8306, 1.8609
]

vgg16_train_loss = [
    1.8706, 1.3462, 1.1527, 1.0039, 0.8651, 0.7485, 0.6333,
    0.5286, 0.4393, 0.3585, 0.2877, 0.2505, 0.2036, 0.1886,
    0.1587, 0.1318, 0.1261, 0.1171, 0.0991, 0.1025
]
vgg16_valid_loss = [
    1.4625, 1.2409, 1.1737, 1.0256, 1.0359, 0.9218, 1.0652,
    0.9870, 0.9557, 1.0587, 1.0799, 1.2065, 1.1997, 1.1818,
    1.2805, 1.2809, 1.3594, 1.2518, 1.4273, 1.3592
]

fig, axs = plt.subplots(2, 1, figsize=(12, 12))

axs[0].plot(epochs, resnet18_train_loss, label='ResNet18 Train Loss', marker='o')
axs[0].plot(epochs, efficientnet_b0_train_loss, label='EfficientNet B0 Train Loss', marker='o')
axs[0].plot(epochs, mobilenet_train_loss, label='MobileNet Train Loss', marker='o')
axs[0].plot(epochs, vgg16_train_loss, label='VGG16 Train Loss', marker='o')

axs[0].set_title('Training Loss for Different Models')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].grid()
axs[0].set_ylim(0, 3)  

axs[1].plot(epochs, resnet18_valid_loss, label='ResNet18 Valid Loss', marker='x')
axs[1].plot(epochs, efficientnet_b0_valid_loss, label='EfficientNet B0 Valid Loss', marker='x')
axs[1].plot(epochs, mobilenet_valid_loss, label='MobileNet Valid Loss', marker='x')
axs[1].plot(epochs, vgg16_valid_loss, label='VGG16 Valid Loss', marker='x')

axs[1].set_title('Validation Loss for Different Models')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].grid()
axs[1].set_ylim(0, 3)  

plt.tight_layout()  
plt.savefig('model_loss_wo_pretrain.pdf', format='pdf', dpi=300)

plt.show()