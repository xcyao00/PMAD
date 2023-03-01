import random
import math
import torch
import numpy as np
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


class BlockWiseMaskGenerator:
    def __init__(
            self, input_size, num_masking_patches, 
            min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.target_to_mask_ratio = {0: 0.42, 1: 0.28, 2: 0.16, 3: 0.16,
                4: 0.16, 5: 0.36, 6: 0.12, 7: 0.48,
                8: 0.48, 9: 0.04, 10: 0.38, 11: 0.22,
                12: 0.48, 13: 0.46, 14: 0.2}

    def __repr__(self):
        repr_str = "BlockWiseMaskGenerator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def __call__(self, target=-1, foreground_mask=None, image=None, sampling_weights=None):
        if target != -1:
            self.num_masking_patches = int(196 * self.target_to_mask_ratio[target])

        if foreground_mask is not None:  # masking at foreground
            fg_mask = patchify(foreground_mask, num_chans=1)
            fg_mask = fg_mask.sum(dim=-1)
            fg_mask[fg_mask != 0] = 1
            
            mask = self._masking(sampling_weights=sampling_weights, image=image, fg_mask=fg_mask)
        else:  # masking at all patches
            mask = self._masking(sampling_weights=sampling_weights, image=image, fg_mask=None)

        return mask

    def _masking(self, sampling_weights=None, image=None, fg_mask=None):
        if sampling_weights is not None:
            max_val = sampling_weights.max()
            min_val = sampling_weights.min()
            sampling_weights = (sampling_weights - min_val) / (max_val - min_val)
        if image is not None:
            freq = torch.rfft(image, 2, onesided=False, normalized=True)
            freq = torch.sqrt(freq[..., 0]**2 + freq[..., 1]**2)
            freq = freq / freq.max()
            freq[torch.isnan(freq)] = 0.0
            freq = torch.clamp(freq, min=0.0, max=1.0)
            freq = patchify(freq)
            freq = freq.sum(-1)
        else:
            freq = None

        mask = torch.zeros(self.height, self.width, dtype=torch.int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches, sampling_weights, freq, fg_mask)
            if delta == 0:
                break
            else:
                mask_count += delta
        
        return mask.reshape(-1)
    
    def _mask(self, mask, max_mask_patches, sampling_weights=None, freq=None, fg_mask=None):
        """
        sampling_weights: Tensor, (196, )
        freq: Tensor, (196, )
        """
        delta = 0
        idxes = np.arange(self.num_patches).reshape(14, 14)
        if fg_mask is not None:
            fg_mask = fg_mask.reshape(14, 14).numpy()
            # print(np.sum(fg_mask))
        if sampling_weights is not None:
            weights = sampling_weights.reshape(14, 14)
            weights = weights.numpy()
        if freq is not None:
            weights = freq.reshape(14, 14)
            weights = weights.numpy()
        
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                if fg_mask is not None:
                    # valid indexes
                    idxes_r = idxes[0:self.height-h, 0:self.width-w]
                    # valid fg masks
                    fg_mask_r = fg_mask[0:self.height-h, 0:self.width-w]
                    # valid indexes
                    if np.sum(fg_mask_r == 1) != 0:
                        idxes_fg_r = idxes_r[fg_mask_r == 1]
                    else:
                        idxes_fg_r = idxes_r.reshape(-1)
                    if sampling_weights is not None or freq is not None:
                        weights_r = weights[0:self.height-h, 0:self.width-w]
                        if np.sum(fg_mask_r == 1) != 0:
                            weights_r = weights_r[fg_mask_r == 1]
                        else:
                            weights_r = weights_r.reshape(-1)
                        weights_r = weights_r / np.sum(weights_r)
                        idx = np.random.choice(idxes_fg_r, 1, replace=False, p=weights_r)
                    else:
                        idx = np.random.choice(idxes_fg_r, 1, replace=False)
                    idx = idx[0]
                    top = idx // 14
                    left = idx % 14
                else:
                    if sampling_weights is not None or freq is not None:
                        # valid indexes
                        idxes_r = idxes[0:self.height-h, 0:self.width-w]
                        weights_r = weights[0:self.height-h, 0:self.width-w]
                        weights_r = weights_r / np.sum(weights_r)
                        idx = np.random.choice(idxes_r.reshape(-1), 1, replace=False, p=weights_r.reshape(-1))

                        idx = idx[0]
                        top = idx // 14
                        left = idx % 14
                    else:
                        top = random.randint(0, self.height - h)
                        left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta


class RandomMaskGenerator:
    def __init__(
        self, input_size, num_masking_patches):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.target_to_mask_ratio = {0: 0.42, 1: 0.28, 2: 0.16, 3: 0.16,
                4: 0.16, 5: 0.36, 6: 0.12, 7: 0.48,
                8: 0.48, 9: 0.04, 10: 0.38, 11: 0.22,
                12: 0.48, 13: 0.46, 14: 0.2}

    def __repr__(self):
        repr_str = "RandomMaskGenerator(%d, %d -> %d)" % (
            self.height, self.width, 
            self.num_masking_patches)
        return repr_str

    def get_shape(self):
        return self.height, self.width
    
    def __call__(self, target=-1, foreground_mask=None, image=None, sampling_weights=None):
        if target != -1:
            self.num_masking_patches = int(196 * self.target_to_mask_ratio[target])

        if foreground_mask is not None:  # masking at foreground
            fg_mask = patchify(foreground_mask, num_chans=1)
            fg_mask = fg_mask.sum(dim=-1)
            fg_mask[fg_mask != 0] = 1
            num_patches = int(fg_mask.sum())
            if image is not None:  # masking by frequency
                mask0 = self._masking_by_frequency(image, num_patches, fg_mask)
                mask = torch.zeros(self.num_patches, dtype=torch.int)
                mask[fg_mask != 0] = mask0
            elif sampling_weights is not None: # masking by weights
                mask0 = self._masking_by_weights(sampling_weights, num_patches, fg_mask)
                mask = torch.zeros(self.num_patches, dtype=torch.int)
                mask[fg_mask != 0] = mask0
            else:  # random masking
                mask0 = self._random_masking(num_patches)
                mask = torch.zeros(self.num_patches, dtype=torch.int)
                mask[fg_mask != 0] = mask0
        else:  # masking at all patches
            if image is not None:  # masking by frequency
                mask = self._masking_by_frequency(image, self.num_patches)
            elif sampling_weights is not None: # masking by weights
                mask = self._masking_by_weights(sampling_weights, self.num_patches)
            else:  # random masking
                mask = self._random_masking(self.num_patches)

        return mask
    
    def _masking_by_frequency(self, image, num_patches, fg_mask=None):
        freq = torch.rfft(image, 2, onesided=False, normalized=True)
        freq = torch.sqrt(freq[..., 0]**2 + freq[..., 1]**2)
        freq = freq / freq.max()
        freq[torch.isnan(freq)] = 0.0
        freq = torch.clamp(freq, min=0.0, max=1.0)
        freq = patchify(freq)
        freq = freq.sum(-1)

        if fg_mask is not None:
            weights = freq[fg_mask != 0]  # only at foreground 
        else:
            weights = freq
        max_val = weights.max()
        min_val = weights.min()
        weights = (weights - min_val) / (max_val - min_val)
        weights = weights / weights.sum()
        weights = weights.numpy()
        idxes = np.arange(num_patches)
        mask_idxes = np.random.choice(idxes, self.num_masking_patches, replace=True, p=weights)
        mask = np.zeros(num_patches, dtype=np.int32)
        mask[mask_idxes] = 1
        mask = torch.from_numpy(mask)

        return mask

    def _masking_by_weights(self, sampling_weights, num_patches, fg_mask=None):
        if fg_mask is not None:
            weights = sampling_weights[fg_mask != 0]  # only at foreground 
        else:
            weights = sampling_weights
        weights[weights == 0] = 1  # higher weights for non masked patches
        max_val = weights.max()
        min_val = weights.min()
        weights = (weights - min_val) / (max_val - min_val)
        weights = weights / weights.sum()
        weights = weights.numpy()
        idxes = np.arange(num_patches)
        mask_idxes = np.random.choice(idxes, self.num_masking_patches, replace=True, p=weights)
        mask = np.zeros(num_patches, dtype=np.int32)
        mask[mask_idxes] = 1
        mask = torch.from_numpy(mask)

        return mask

    def _random_masking(self, num_patches):
        len_keep = num_patches - self.num_masking_patches
                
        noise = torch.rand(num_patches)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones(num_patches, dtype=torch.int)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore).squeeze()

        return mask

class MaskGenerator:
    """
    A warp class for generating masks.
    """
    def __init__(self, args):
        self.block_wise_mask_generator = BlockWiseMaskGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )
        self.random_mask_generator = RandomMaskGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches
        )
    
    def __call__(self, target, foreground_mask=None, image=None, sampling_weights=None):
        n1 = random.random() 
        if n1 < 0.5:  # generating masks in foreground regions
            n2 = np.random.choice([0, 1, 2], 1, replace=True)
            n3 = random.random()
            if n2[0] == 0:  # frequency masking
                if n3 < 0.5:  # blockwise mask
                    mask = self.block_wise_mask_generator(target=target, foreground_mask=foreground_mask, image=image)
                else:
                    mask = self.random_mask_generator(target=target, foreground_mask=foreground_mask, image=image)
            elif n2[0] == 1:
                if n3 < 0.5:
                    mask = self.block_wise_mask_generator(target=target, foreground_mask=foreground_mask, sampling_weights=sampling_weights)
                else:
                    mask = self.random_mask_generator(target=target, foreground_mask=foreground_mask, sampling_weights=sampling_weights)
            else:
                if n3 < 0.5:
                    mask = self.block_wise_mask_generator(target=target, foreground_mask=foreground_mask)
                else:
                    mask = self.random_mask_generator(target=target, foreground_mask=foreground_mask)
        else:  # generating masks in all regions
            n2 = np.random.choice([0, 1, 2], 1, replace=True)
            n3 = random.random()
            if n2[0] == 0:  # frequency masking
                if n3 < 0.5:
                    mask = self.block_wise_mask_generator(target=target, image=image)
                else:
                    mask = self.random_mask_generator(target=target, image=image)
            elif n2[0] == 1:
                if n3 < 0.5:
                    mask = self.block_wise_mask_generator(target=target, sampling_weights=sampling_weights)
                else:
                    mask = self.random_mask_generator(target=target, sampling_weights=sampling_weights)
            else:
                if n3 < 0.5:
                    mask = self.block_wise_mask_generator(target=target)
                else:
                    mask = self.random_mask_generator(target=target)
        
        if sampling_weights is not None:
            mask_weights = self._mask_weighting(mask, sampling_weights)
            return mask, mask_weights
            
        return mask
    
    def _mask_weighting(self, mask, sampling_weights, bias=1, k=2):
        masked_weights = sampling_weights[mask == 1]
        ind = torch.argsort(masked_weights)
        rank = torch.argsort(ind)
        rank_weights = rank / (rank.shape[0] - 1)
        mask_weights = (bias + rank_weights * (2 - bias)).pow(k)

        return mask_weights


def patchify(imgs, patch_size=16, num_chans=3):
    """
    Convert an image into image patches.
    
    Args:
        imgs: shape of (N, 3, H, W) or (N, 1, H, W).
    Returns:
        patches: (N, L, patch_size**2 * 3) or (N, L, patch_size**2 * 1).
    """
    assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % patch_size == 0

    h = w = imgs.shape[1] // patch_size
    patches = imgs.reshape(shape=(num_chans, h, patch_size, w, patch_size))
    patches = torch.einsum('chpwq->hwpqc', patches)
    patches = patches.reshape(shape=(h * w, patch_size**2 * num_chans))
    
    return patches