import matplotlib.pyplot as plt

epochs = list(range(1, 21))

resnet18_valid_accuracy = [0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]

efficientnet_b0_valid_accuracy = [0.50, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.81, 0.83, 0.84, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]

mobilenet_valid_accuracy = [0.48, 0.58, 0.63, 0.67, 0.70, 0.73, 0.76, 0.78, 0.80, 0.82, 0.83, 0.84, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]

vgg16_valid_accuracy = [0.52, 0.62, 0.67, 0.70, 0.72, 0.75, 0.78, 0.80, 0.81, 0.83, 0.84, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94]

plt.figure(figsize=(12, 6))

plt.plot(epochs, resnet18_valid_accuracy, label='ResNet18 Valid Accuracy', marker='o')
plt.plot(epochs, efficientnet_b0_valid_accuracy, label='EfficientNet B0 Valid Accuracy', marker='o')
plt.plot(epochs, mobilenet_valid_accuracy, label='MobileNet Valid Accuracy', marker='o')
plt.plot(epochs, vgg16_valid_accuracy, label='VGG16 Valid Accuracy', marker='o')

plt.title('Validation Accuracy for Different Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.ylim(0, 1)  

plt.tight_layout()  
plt.savefig('model_accuracy_plot_wo_pretrain.pdf', format='pdf', dpi=300)

plt.show()