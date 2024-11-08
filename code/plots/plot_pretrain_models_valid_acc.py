import matplotlib.pyplot as plt

epochs = list(range(1, 21))

resnet18_valid_accuracy = [
    0.7477, 0.7772, 0.7909, 0.7982, 0.7998, 0.8027,
    0.8003, 0.8042, 0.8090, 0.8063, 0.8027, 0.8050,
    0.8046, 0.8109, 0.8132, 0.8066, 0.8088, 0.8110,
    0.8214, 0.8147
]

efficientnet_valid_accuracy = [
    0.5918, 0.6851, 0.7250, 0.7455, 0.7666, 0.7746,
    0.7870, 0.7913, 0.7941, 0.7958, 0.8018, 0.8081,
    0.8057, 0.8039, 0.8089, 0.8077, 0.8074, 0.8126,
    0.8125, 0.8112
]

mobilenet_valid_accuracy = [
    0.6895, 0.7327, 0.7571, 0.7767, 0.7790, 0.7823,
    0.7843, 0.7861, 0.7938, 0.7955, 0.7932, 0.8053,
    0.7947, 0.8007, 0.7948, 0.7998, 0.7987, 0.8068,
    0.8047, 0.8057
]

vgg16_valid_accuracy = [
    0.7540, 0.8016, 0.8247, 0.8444, 0.8491, 0.8364,
    0.8653, 0.8536, 0.8573, 0.8547, 0.8578, 0.8586,
    0.8510, 0.8620, 0.8598, 0.8631, 0.8644, 0.8588,
    0.8543, 0.8658
]


plt.figure(figsize=(10, 10))

plt.plot(epochs, resnet18_valid_accuracy, label='ResNet18 Valid Accuracy', marker='o')
plt.plot(epochs, efficientnet_valid_accuracy, label='EfficientNet Valid Accuracy', marker='o')
plt.plot(epochs, mobilenet_valid_accuracy, label='MobileNet Valid Accuracy', marker='o')
plt.plot(epochs, vgg16_valid_accuracy, label='VGG16 Valid Accuracy', marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.savefig('model_accuracy_plot.pdf', format='pdf', dpi=300)
plt.show()