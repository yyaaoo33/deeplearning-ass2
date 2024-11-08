import matplotlib.pyplot as plt

epochs = list(range(1, 21))

# ResNet18
resnet18_train_loss = [
    1.0760, 0.6185, 0.4346, 0.3051, 0.2203, 0.1603,
    0.1374, 0.1091, 0.0945, 0.0810, 0.0706, 0.0715,
    0.0659, 0.0598, 0.0586, 0.0564, 0.0472, 0.0488,
    0.0459, 0.0399
]

resnet18_valid_loss = [
    0.7426, 0.6356, 0.6304, 0.6470, 0.6981, 0.7703,
    0.7712, 0.8206, 0.8325, 0.8426, 0.8739, 0.8531,
    0.8823, 0.8755, 0.8680, 0.9064, 0.9207, 0.9094,
    0.8858, 0.9237
]

# EfficientNet
efficientnet_train_loss = [
    1.6637, 1.0768, 0.8663, 0.7411, 0.6448, 0.5676,
    0.4918, 0.4336, 0.3833, 0.3418, 0.3072, 0.2640,
    0.2297, 0.2102, 0.1850, 0.1680, 0.1520, 0.1384,
    0.1265, 0.1149
]

efficientnet_valid_loss = [
    1.1665, 0.9113, 0.7947, 0.7296, 0.6729, 0.6468,
    0.6319, 0.6238, 0.6192, 0.6219, 0.6272, 0.6352,
    0.6509, 0.6586, 0.6767, 0.7035, 0.7121, 0.7107,
    0.7332, 0.7426
]

# MobileNet
mobilenet_train_loss = [
    1.1985, 0.8199, 0.6609, 0.5533, 0.4515, 0.3839,
    0.3276, 0.2717, 0.2327, 0.1959, 0.1745, 0.1524,
    0.1345, 0.1339, 0.1156, 0.1103, 0.0989, 0.0962,
    0.0962, 0.0833
]

mobilenet_valid_loss = [
    0.8779, 0.7538, 0.6902, 0.6545, 0.6450, 0.6508,
    0.6621, 0.6954, 0.7012, 0.7204, 0.7485, 0.7469,
    0.8305, 0.8173, 0.8498, 0.8291, 0.8697, 0.8617,
    0.8555, 0.8705
]

# VGG16
vgg16_train_loss = [
    1.2833, 0.6116, 0.3988, 0.2730, 0.1959, 0.1400,
    0.1186, 0.0990, 0.0913, 0.0745, 0.0687, 0.0558,
    0.0697, 0.0623, 0.0559, 0.0583, 0.0479, 0.0549,
    0.0440, 0.0427
]

vgg16_valid_loss = [
    0.7482, 0.5991, 0.5554, 0.5478, 0.5080, 0.6059,
    0.6441, 0.6604, 0.6191, 0.6471, 0.6764, 0.6654,
    0.9136, 0.7054, 0.6944, 0.7628, 0.8114, 0.6442,
    0.9250, 0.7975
]

plt.figure(figsize=(10, 20))

# Train Loss
plt.subplot(2, 1, 1)
plt.plot(epochs, resnet18_train_loss, label='ResNet18 Train Loss', marker='o')
plt.plot(epochs, efficientnet_train_loss, label='EfficientNet Train Loss', marker='o')
plt.plot(epochs, mobilenet_train_loss, label='MobileNet Train Loss', marker='o')
plt.plot(epochs, vgg16_train_loss, label='VGG16 Train Loss', marker='o')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Valid Loss
plt.subplot(2, 1, 2)
plt.plot(epochs, resnet18_valid_loss, label='ResNet18 Valid Loss', marker='o')
plt.plot(epochs, efficientnet_valid_loss, label='EfficientNet Valid Loss', marker='o')
plt.plot(epochs, mobilenet_valid_loss, label='MobileNet Valid Loss', marker='o')
plt.plot(epochs, vgg16_valid_loss, label='VGG16 Valid Loss', marker='o')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig('model_loss_plot.pdf', format='pdf', dpi=300)
plt.show()
