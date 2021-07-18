from torchvision import transforms

def raw_transform():

    # CenterCrop

    # RandomCrop
    # transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')

    # RandomResizedCrop
    # transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation)

    # FiveCrop, TenCrop
    # transforms.FiveCrop(size)
    # transforms.TenCrop(size, vertical_flip=False)

    # RandomHorizontalFlip, RandomVerticalFlip

    # RandomRotation
    # transforms.RandomRotation(degrees, resample=False, expand=False, center=None)

    # Pad
    # transforms.Pad(padding, fill=0, padding_mode='constant')

    # ColorJitter
    # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

    # Grayscale, RandomGrayscale
    # transforms.Grayscale(num_output_channels)
    # transforms.RandomGrayscale(num_output_channels, p=0.1)

    # RandomAffine
    # transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)

    # LinearTransformation

    # RandomErasing: operate on tensor
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

    # Lambda
    # tranforms.Lambda(lambd)

    # Resize

    # ToTensor

    # Normalize: output = (input - mean) / std
    # transforms.Normalize(mean, std, inplace=False)


def apply_transform():
    # RandomChoice
    # transforms.RandomChoice([transforms1, transforms2, transforms3])

    # RandomApply
    # transforms.RandomApply([transforms1, transforms2, transforms3], p=0.5)

    # RandomOrder
    # transforms.RandomOrder([transforms1, transforms2, transforms3])

    # Compose


def custom_transform():
    """
    class YourTransforms(object):
        def __init__(self, ...):
            ...
        def __call__(self, img):
            ...
            return img
    """

class AddPepperNoise(object):
    """
    Args:
        snr (float): signal noise rate
        p (float): probility
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or isinstance(p, float)
        self.snr = snr
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            img (PIL Image): PIL Image
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            singal_pct = self.snr
            noise_pat = 1 - self.snr
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255
            img_[mask == 2] = 0
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img
