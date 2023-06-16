import matplotlib.pyplot as plt



def return_dataset_images(train_loader,total_images):
  dataiter = iter(train_loader)
  images, labels = next(dataiter)
  figure = plt.figure()
  for index in range(1, total_images + 1):
      plt.subplot(6, 10, index)
      plt.axis('off')
      plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
