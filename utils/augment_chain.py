from torchvision import transforms
from .doppler_augmentation import DopplerMask, DopplerScale, DopplerShift
from .temporal_augmentation import TemporalDrop, RandomTemporalCrop, TemporalPad, TemporalScale, TemporalMask
from .random_augmentation import OneOf, UseWithProb

size = 224

temporal_crop = transforms.Compose([RandomTemporalCrop(0.9),
                                   OneOf([TemporalPad(size),
                                          TemporalScale(size)])])

temporal_drop = transforms.Compose([TemporalDrop(0.9),
                                   TemporalScale(size)])

temporal_aug = OneOf([temporal_crop, temporal_drop])

spec_mask = transforms.Compose([UseWithProb(TemporalMask()), UseWithProb(DopplerMask())])

# im_aug = DopplerShift(ratio=-0.1)(im_np_gray)
# im_aug = DopplerScale(ratio=1.2)(im_np_gray)
# im_aug = AugMix(transform_list)(im_np_gray)
# im_aug = spec_mask(im_np_gray) # input gray scale image
# im_aug = RandomTemporalCrop(crop_ratio=0.9)(im_np_gray)
# im_aug = TemporalPad(size=im_np_gray.shape[1])(im_aug)
# im_aug = UseWithProb(TemporalDrop(0.9))(im_np_gray)
# im_aug = TemporalScale(size=im_np_gray.shape[1])(im_aug)
# im_aug = OneOf([TemporalDrop(0.9), SpecAugment()])(im_np_gray)
# im_aug = temporal_drop(im_np_gray)
# trace_x, trace_y = dtw(im_aug.transpose(), im_np_gray.transpose())
# x_ind = np.linspace(0, im_aug.shape[1]-1,im_aug.shape[1]).astype(int)
# y_ind = np.interp(x_ind, trace_x, trace_y).astype(int)
# mask = np.zeros([im_aug.shape[1],im_aug.shape[1]])
# mask[y_ind, x_ind] = 1
# im_scaled = np.matmul(im_aug, mask)
# im_np2pil = Image.fromarray(im_scaled)
# f, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax2.imshow(mask, cmap='hot', interpolation='nearest')
# ax3.imshow(im_np2pil, cmap="gray")
# ax1.plot(x_ind, y_ind,'rx')

