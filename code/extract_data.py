import tarfile

tar_file_path = 'cifar-10-python.tar.gz'
extract_path = './data'

with tarfile.open(tar_file_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)