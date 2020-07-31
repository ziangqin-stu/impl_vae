# import torch
# import torchvision
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
#
# transform = transforms.Compose([transforms.ToTensor()])
# mnist_train = datasets.MNIST('./data/mnist', download=True, transform=transform, train=True)
# mnist_test = datasets.MNIST('./data/mnist', download=True, transform=transform, train=False)
# dataloader_train = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=100, shuffle=True)
# dataloader_test = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=100, shuffle=True)
#
# images, lables = next(iter(dataloader_train))
# plt.imshow(images[0].reshape(28,28), cmap="gray")