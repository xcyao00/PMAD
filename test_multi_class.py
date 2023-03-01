import warnings
warnings.filterwarnings('ignore')
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from datasets.mvtec_test import MVTEC
from protoflow.utils import measure_rocaucs
from protoflow.protoflow import ProtoFlow
from utils.mask_generator import RandomMaskGenerator
from timm.models import create_model
import utils.utils as utils
import reconstruction.models
import vit_tokenizer.tokenizer as tokenizer


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
xent_loss = nn.CrossEntropyLoss(reduction='none')


def prepare_model(args, ckpt_dir):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
    )
    checkpoint = torch.load(ckpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    return model


def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained=True,
            pretrained_weight=args.tokenizer_weight_path,
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model


def generate_mask(mask, mask_generator, label=None):
    if label == 0:
        mask = mask_generator()
        mask = mask.to(mask.device)
        return mask
    else:
        return mask


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def run_one_image(img, mask, model, labels):
    # run patch-level reconstruction model
    pred_t = model(img.float(), mask)

    uncertainty = xent_loss(input=pred_t, target=labels)
    a_map = torch.zeros(1, 196).to(img.device)

    a_map[mask == 1] = uncertainty
    a_map[mask == 0] = 0  # unmask patch, no anomaly score
    a_map = a_map.reshape(1, 14, 14) 
    a_map = F.interpolate(a_map.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=True).squeeze()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    mask = mask.to(img.dtype)

    x = torch.einsum('nchw->nhwc', img).detach().cpu()
   
    # masked image
    im_masked = x * (1 - mask)

    ori_img = torch.clip((x[0] * imagenet_std + imagenet_mean), 0, 255)
    msk_img = torch.clip((im_masked[0] * imagenet_std + imagenet_mean), 0, 255)
    a_map = a_map.cpu()

    return ori_img, msk_img, a_map


def evaluation(args, model, tokenizer, protoflow):
    device = torch.device(args.device)
    args.window_size = (args.input_size // 16, args.input_size // 16)

    dataset = MVTEC(args.data_path, class_name=args.class_name, train=False,
                    norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225], img_size=224, crp_size=224, tokenizer_model=args.tokenizer_model)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    
    image_aucs, pixel_aucs = [], []
    pre_image_aucs, pre_pixel_aucs = [], []
    for mask_ratio in np.arange(0.02, 0.5, 0.02).tolist():
        mask_generator = RandomMaskGenerator(
            args.window_size, num_masking_patches=int(196*mask_ratio))
        pre_masks, img_labels, abnomalities = protoflow.predict(data_loader, args.class_name, mask_ratio)

        gt_label_list = []
        gt_mask_list = []
        score_list = []
        img_types = []
        ori_img_list, msk_img_list = [], []
        for i, (img, token_img, label, mask, _, img_type) in enumerate(tqdm(data_loader, 'Evaluation Dataset->{}.{}'.format('MVTEC', args.class_name))):
            gt_label_list.append(label.cpu().numpy())
            gt_mask_list.append(mask.squeeze().cpu().numpy())
            img_types.append(img_type)

            img = img.to(device, non_blocking=True)
            token_img = token_img.to(device, non_blocking=True)
            with torch.no_grad():
                mask = pre_masks[i]
                mask = torch.from_numpy(mask).to(device).to(torch.long)
                mask = mask.unsqueeze(0)
                img_label = img_labels[i]
                mask = generate_mask(mask, mask_generator, label=img_label)

                input_ids = tokenizer.get_codebook_indices(token_img).flatten(1)
                mask = mask.reshape(1, 196).to(torch.bool)
                labels = input_ids[mask]

                ori_img, msk_img, score = run_one_image(img, mask, model, labels)
                score = score.numpy()
                score = score * abnomalities[i]
                ori_img_list.append(ori_img)
                msk_img_list.append(msk_img)
                score = gaussian_filter(score, sigma=4)
            score_list.append(score)

        scores = np.stack(score_list, axis=0)
        labels = np.asarray(gt_label_list)
        masks = np.asarray(gt_mask_list, dtype=np.bool)
        img_scores = np.max(scores.reshape(scores.shape[0], -1), axis=-1)
        image_auc, pixel_auc, _ = measure_rocaucs(labels, masks, img_scores, scores)
        print(f'Category {args.class_name}: Image-AUC: {image_auc}, Pixel AUC: {pixel_auc}, Mask Ratio: {mask_ratio}')
        image_aucs.append(image_auc)
        pixel_aucs.append(pixel_auc)
    
    return np.max(image_aucs), np.max(pixel_aucs), np.arange(0.02, 0.5, 0.02).tolist()[np.argmax(image_aucs)]


def parse_args():
    parser = argparse.ArgumentParser(description='One-for-All: Proposal Masked Cross-Class Anomaly Detection')
    # Model parameters
    parser.add_argument('--model_path', type=str,
        default='output_dir/vit_base_16_checkpoint_962_955.pth', help='checkpoint path of model')
    parser.add_argument('--model', default='vit_base_patch16_224_8k_vocab', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")
    
    # Tokenizer settings
    parser.add_argument("--tokenizer_weight_path", type=str, default='weights/tokenizer/vit_tokenizer.pth')  # tokenizer/vit_tokenizer.pth or tokenizer/
    parser.add_argument("--tokenizer_model", type=str, default="vit_tokenizer")  # dall-e or vit_tokenizer
    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='number of codebook')

    # Dataset parameters
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument('--class_name', default='bottle')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.3, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument('--vis', type=eval, choices=[True, False], default=False)
    parser.add_argument("--save_path", type=str, default="./mvtec_results")
    
    parser.add_argument('--second_input_size', default=224, type=int,
                    help='images input size for discrete vae')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=4)
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    model = prepare_model(args, args.model_path)
    print('Model loaded.')
    model.to(device)
    
    if args.tokenizer_model == 'dall-e':
        args.second_input_size = 112
        tokenizer = utils.create_d_vae(
            weight_path=args.tokenizer_weight_path, d_vae_type=args.tokenizer_model,
            device=device, image_size=args.second_input_size)
    else:
        tokenizer = get_visual_tokenizer(args).to(device)
    
    protoflow = ProtoFlow('weights/protoflow/weights_multi_class.pt')
    
    all_image_aucs, all_pixel_aucs, mask_ratios = [], [], []
    for class_name in MVTEC.CLASS_NAMES:
        args.class_name = class_name

        image_auc, pixel_auc, mask_ratio = evaluation(args, model, tokenizer, protoflow)
        print('Category: {}'.format(class_name))
        print("Max Image-AUC: {}".format(image_auc))
        print("Max Pixel-AUC: {}".format(pixel_auc))
        all_image_aucs.append(image_auc)
        all_pixel_aucs.append(pixel_auc)
        mask_ratios.append(mask_ratio)
    
    for i, class_name in enumerate(MVTEC.CLASS_NAMES):
        print(f'{class_name}: Image-AUC: {all_image_aucs[i]}, Pixel-AUC: {all_pixel_aucs[i]}, Mask-Ratio: {mask_ratios[i]}')
    print('Mean Image-AUC: {}'.format(np.mean(all_image_aucs)))
    print('Mean Pixel-AUC: {}'.format(np.mean(all_pixel_aucs)))
    

if __name__ == '__main__':
    main()

