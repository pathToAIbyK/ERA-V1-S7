import matplotlib.pyplot as plt

def return_dataset_images():
  figure = plt.figure()
  num_of_images = 60
  for index in range(1, num_of_images + 1):
      plt.subplot(6, 10, index)
      plt.axis('off')
      plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
