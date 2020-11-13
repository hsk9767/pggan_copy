# pggan_copy
cpoied from 'https://github.com/jeromerony/Progressive_Growing_of_GANs-PyTorch'

## pg gan

- Contribution

  Used cutmix-like method for discrimination training. 
  When training discriminator, fake images were randomly(of probability 1/4) replaced with real image. 
  As the replaced image, named as 'mixed_fake_img', if filled with real(1/4) & fake(3/4), the output value of discriminaator will be multiplied (1 - 1/4).
  Like this way, the discriminator can be trained to recognize 'full-fake-image', 'quarter-fake-image'.
  
  
- Generated Images(64 images of size 256)
![fake_images-0051-p6 00](https://user-images.githubusercontent.com/34028332/99072036-96f8c380-25f6-11eb-9d0f-1be4b0727eb7.png)


