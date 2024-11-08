import pickle
import numpy as np

def load_cifar_batch(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        data = data_dict[b'data']
        labels = data_dict[b'labels']
        data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(labels)
        return data, labels

def load_cifar10(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 5):  
        data_batch, labels_batch = load_cifar_batch(f'{data_dir}/data_batch_{i}')
        train_data.append(data_batch)
        train_labels.append(labels_batch)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    val_data, val_labels = load_cifar_batch(f'{data_dir}/data_batch_5')  
    test_data, test_labels = load_cifar_batch(f'{data_dir}/test_batch')

    return train_data, train_labels, val_data, val_labels, test_data, test_labels