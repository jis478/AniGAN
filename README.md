# AniGAN
#### AniGAN (https://arxiv.org/pdf/2102.12593.pdf) implementation

This code seems to understand the structure & texture information provided by training images, but doesn't produce meaningful results yet. I will keep debugging and posting the results here.

![image](https://user-images.githubusercontent.com/19499513/114551562-8f6d4180-9c9e-11eb-88e3-f6953ab70e95.png)

## Implementation details

- [x] AST block
- [x] FST blocks 
- [x] Style encoder
- [x] Content encoder
- [x] Double branch discriminator
- [x] POLIN
- [x] AdaPOLIN
- [x] adv loss, reconstruction loss, feature matching loss (hook on activations)
