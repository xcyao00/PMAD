import os
import os.path
import random
from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset

import torch
from random import random
from torchvision import transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from dall_e.utils import map_pixels
from utils.mask_generator import MaskGenerator


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        while True:
            try:
                path, target = self.samples[index]
                fg_path = path.replace('train', 'fg_mask')
                sample = self.loader(path)
                fg_mask = Image.open(fg_path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            sample = self.transform(sample, fg_mask, target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


class ImageMaskPreprocess(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        
        self.target_to_class_name = {0: 'bottle', 1: 'cable', 2: 'capsule', 3: 'carpet',
                            4: 'grid', 5: 'hazelnut', 6: 'leather', 7: 'metal_nut',
                            8: 'pill', 9: 'screw', 10: 'tile', 11: 'toothbrush',
                            12: 'transistor', 13: 'wood', 14: 'zipper'}
        self.sampling_weights = {'bottle': torch.zeros(196), 'cable': torch.zeros(196), 'capsule': torch.zeros(196), 'carpet': torch.zeros(196), 'grid': torch.zeros(196),
                                 'hazelnut': torch.zeros(196), 'leather': torch.zeros(196), 'metal_nut': torch.zeros(196), 'pill': torch.zeros(196), 'screw': torch.zeros(196),
                                 'tile': torch.zeros(196), 'toothbrush': torch.zeros(196), 'transistor': torch.zeros(196), 'wood': torch.zeros(196), 'zipper': torch.zeros(196)}
        self.sampling_counts = {'bottle': torch.zeros(196), 'cable': torch.zeros(196), 'capsule': torch.zeros(196), 'carpet': torch.zeros(196), 'grid': torch.zeros(196),
                                 'hazelnut': torch.zeros(196), 'leather': torch.zeros(196), 'metal_nut': torch.zeros(196), 'pill': torch.zeros(196), 'screw': torch.zeros(196),
                                 'tile': torch.zeros(196), 'toothbrush': torch.zeros(196), 'transistor': torch.zeros(196), 'wood': torch.zeros(196), 'zipper': torch.zeros(196)}
        self.sampling_by_weights = False

        self.image_transform = transforms.Compose([
            transforms.Resize(args.input_size, Image.ANTIALIAS),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))])
        self.mask_transform = transforms.Compose([
            transforms.Resize(args.input_size, Image.NEAREST),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor()])

        if args.tokenizer_model == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.Resize(args.second_input_size, Image.ANTIALIAS),
                transforms.CenterCrop(args.second_input_size),
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.tokenizer_model == "vit_tokenizer":
            self.visual_token_transform = transforms.Compose([
                transforms.Resize(args.second_input_size, Image.ANTIALIAS),
                transforms.CenterCrop(args.second_input_size),
                transforms.ToTensor()])
        else:
            raise NotImplementedError()

        self.mask_generator = MaskGenerator(args)

    def __call__(self, image, foreground_mask, target):
        foreground_mask = self.mask_transform(foreground_mask)
        token_image = self.visual_token_transform(image)
        image = self.image_transform(image)

        class_name = self.target_to_class_name[target]
        if self.sampling_by_weights:
            sampling_weights = self.sampling_weights[class_name]
            mask, mask_weights = self.mask_generator(target, foreground_mask, image, sampling_weights)
            return image, token_image, mask, mask_weights
        else:
            sampling_weights = None
            mask = self.mask_generator(target, foreground_mask, image, sampling_weights)
            return image, token_image, mask

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  image_transform = %s,\n" % str(self.image_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr


def build_training_dataset(args):
    transform = ImageMaskPreprocess(args)
    
    return ImageFolder(args.data_path, transform=transform)

