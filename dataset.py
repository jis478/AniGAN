import glob
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch 

class ImageDataset(Dataset):
    def __init__(self, root, size, unpaired=True, mode="train"):
        self.size = size
        self.mode = mode
#         self.files_A = glob.glob(os.path.join(root, 'A', "%s" % mode) + "/*.*")
#         self.files_B = glob.glob(os.path.join(root, 'B', "%s" % mode) + "/*.*")
        self.files_A = glob.glob(os.path.join(root, "%s" % mode, 'A') + "/*.*")
        self.files_B = glob.glob(os.path.join(root, "%s" % mode, 'B') + "/*.*")

        if unpaired:
            random.shuffle(self.files_A)
            random.shuffle(self.files_B)
        assert len(self.files_A) == len(self.files_B), "The number of images are different for domain A/B"

        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.size, Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.size, Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        image_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        image_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        item = {'images_A': image_A, 'images_B': image_B}
        return item

    def __len__(self):
        return len(self.files_A)



# batch image 처리
def patch_generation(images, num_ref=1, num_crops=8, max_size=1/4, min_size=1/8, resize=1/4):
    ''' image가 nchw (1x3x512x512)로 들어온다고 가정 '''
    ''' 총 n개의 patch 사이즈를 생성 후, batch 내에서 patch 사이즈별 patch 생성 (즉, batch 내 이미지 마다 patch 사이즈 별 위치가 같음) '''

    patches = []
    resize = int(images.size(2) * resize)
    batch, c, h, w = images.size()
    # print('hmin', int(h * min_size))
    # print('hmax', int(h * max_size))
    # print('n', num_crops*num_ref)
    crop_size = random.sample(range(int(h * min_size), int(h * max_size)), num_crops * num_ref)

    for size in crop_size:
        pos_h = random.sample(range(0, h - size), 1)[0]
        pos_w = random.sample(range(0, w - size), 1)[0]
        patch = images[:, :, pos_h:pos_h + size, pos_w:pos_w + size]                                            # 20x3x32x 32
        patch = F.interpolate(patch, size=(resize, resize), mode='bilinear', align_corners=False)               # 20x3x128x128
        patches.append(patch)                                                                                   # [20x3x128x128, 20x3x128x128, ... ]

    patches = torch.stack(patches, dim=1)                                                                       # 20x32x3x128x128 (ref) or 20x8x3x128x128
    patches = patches.view(batch, num_crops, num_ref, c, resize, resize)                                                  # 20x8x4x3x128x128 (ref) or 20x8x1x3x128x128

    return patches

# # batch image 처리
# def patch_generation(images, patch_num=5, min_size=1/4, max_size=1/8, resize=128):
#     ''' image가 nchw (1x3x512x512)로 들어온다고 가정 '''
#     ''' 하나의 이미지는 하나의 패치 사이즈를 가짐. 5개의 x,y 사이즈가 다른 패치 생성'''
#     patches = []
#     size = random.sample(range(image.size[2]*min_size, image.size[2]*max_size), 2)  # h,w patch size
#     pos_x = random.sample(range(0, image.size[2] - size[0]), patch_num)             # h - h_patch
#     pos_y = random.sample(range(0, image.size[3] - size[1]), patch_num)             # w - w_patch
#     for x, y in zip(pos_x, pos_y):
#         patch = image[x:(x+size[0]), y:(y+size[1]),:]
#         patch = F.interpolate(patch, size=(resize, resize), mode='bilinear', align_corners=False)
#         patches.append(patch)
#     return torch.stack(patches, dim=0)


# single image 처리
# def patch_generation(image, patch_num=5, min_size=1/4, max_size=1/8, resize=128):
#     ''' image가 nchw (1x3x512x512)로 들어온다고 가정 '''
#     ''' 하나의 이미지는 하나의 패치 사이즈를 가짐. 5개의 x,y 사이즈가 다른 패치 생성'''
#     patches = []
#     size = random.sample(range(image.size[2]*min_size, image.size[2]*max_size), 2)  # h,w patch size
#     pos_x = random.sample(range(0, image.size[2] - size[0]), patch_num)             # h - h_patch
#     pos_y = random.sample(range(0, image.size[3] - size[1]), patch_num)             # w - w_patch
#     for x, y in zip(pos_x, pos_y):
#         patch = image[x:(x+size[0]), y:(y+size[1]),:]
#         patch = F.interpolate(patch, size=(resize, resize), mode='bilinear', align_corners=False)
#         patches.append(patch)
#     return torch.stack(patches, dim=0)


# def patch_generation(images, patch_num=5, min_size=80, max_size=120):
#     ''' image가 nhwc로 들어온다고 가정하고, size는 min보다 크고 이미지 보다 작은 사이즈임 '''
#     ''' 하나의 이미지는 하나의 패치 사이즈를 가짐 '''
#     patches = []
#     patch_sizes = random.sample(range(min_size, max_size), images.shape[0])
#     for patch_size, image in zip(patch_sizes, images):
#         pos_x = random.sample(range(0, image.shape[1] - patch_size), patch_num)
#         pos_y = random.sample(range(0, image.shape[2] - patch_size), patch_num)
#         patch = image[pos_x:(pos_x+patch_size), pos_y:(pos_y+patch_size),:]
#         patches.append(patch)  # 이미지가 5장이면 패치도 5개임 (각각 다른 사이즈 패치 임)
#     return patches

