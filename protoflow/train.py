import os
import time
import math
import warnings
import argparse
import numpy as np
import timm
import faiss
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datasets.mvtec_test import MVTEC
from .modules import get_logp, load_decoder_arch, positionalencoding2d
from .utils import embedding_concat, adjust_learning_rate, warmup_learning_rate, load_weights


gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()
idx_to_class = {0: 'bottle', 1: 'cable', 2: 'capsule', 3: 'carpet',
                4: 'grid', 5: 'hazelnut', 6: 'leather', 7: 'metal_nut',
                8: 'pill', 9: 'screw', 10: 'tile', 11: 'toothbrush',
                12: 'transistor', 13: 'wood', 14: 'zipper'}
coreset_bank = {}
base_path = 'weights/prototypes'
for class_name in MVTEC.CLASS_NAMES:
    embedding_path = os.path.join(base_path, class_name)
    coreset = np.load(os.path.join(embedding_path, 'coreset.npy'))
    coreset_bank[class_name] = coreset

# from objects to textures
SEEN_TEXTURES = ['bottle', 'cable', 'capsule', 'screw', 'transistor', 'hazelnut', 'metal_nut', 'pill', 'toothbrush', 'zipper']
UNSEEN_TEXTURES = ['carpet', 'leather', 'grid', 'tile', 'wood']
# from textures to objects
SEEN_OBJECTS = ['carpet', 'leather', 'grid', 'tile', 'wood']
UNSEEN_OBJECTS = ['bottle', 'cable', 'capsule', 'screw', 'transistor', 'hazelnut', 'metal_nut', 'pill', 'toothbrush', 'zipper']
SEEN_CLASSES = MVTEC.CLASS_NAMES
UNSEEN_CLASSES = MVTEC.CLASS_NAMES


def train_external_decoder(c, decoder, e_pair, c_r):
    # score_patches: (784, 9)
    C = e_pair.shape[1]

    if 'cflow' in c.dec_arch:
        z, log_jac_det = decoder(e_pair, [c_r, ])
    else:
        z, log_jac_det = decoder(e_pair)
    
    decoder_log_prob = get_logp(C, z, log_jac_det)
    log_prob = decoder_log_prob / C  # likelihood per dim (256, )
    loss = -log_theta(log_prob).mean()  # (256, )
    
    return loss


def test_external_decoder(c, decoder, index, e_r, c_r, n_neighbors=1, size=32):
    # score_patches: (784, 9)
    C = e_r.shape[1]
    index, coreset = index
    e_r_np = e_r.cpu().numpy()

    _, idx = index.search(e_r_np , k=n_neighbors)
    index_feats = coreset[idx[:, 0]]
    index_feats = torch.from_numpy(index_feats).to(c.device)

    e_pair = e_r - index_feats
    if 'cflow' in c.dec_arch:
        z, log_jac_det = decoder(e_pair, [c_r, ])
    else:
        z, log_jac_det = decoder(e_pair)
    
    decoder_log_prob = get_logp(C, z, log_jac_det)
    log_prob = decoder_log_prob / C  # likelihood per dim (256, )
    loss = -log_theta(log_prob).mean()  # (256, )
    log_prob = log_prob.reshape(size, size)
    
    return log_prob, loss


def generate_prototype(embeds, labels):
    """
    Args:
        t_feats: shape (N, H*W, dim)
        labels: shape (N, )
    """
    protos = []
    for e, label in zip(embeds, labels):
        coreset = coreset_bank[idx_to_class[label.item()]]
        index = faiss.IndexFlatL2(e.shape[-1])
        index.add(coreset) 
        
        e_np = e.cpu().numpy()

        _, idx = index.search(e_np , k=1)
        index_feats = coreset[idx[:, 0]]
        index_feats = torch.from_numpy(index_feats).to(e.device)

        p_r = index_feats  # (h*w, dim)
        protos.append(p_r)
    protos = torch.stack(protos, dim=0)

    return protos

def train_meta_epoch(c, epoch, loader, encoder, decoder, optimizer):
    decoder = decoder.train()  # 3
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_count = 0
        for i, (data) in enumerate(tqdm(loader)):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
            
            image, label = data
            # encoder prediction
            image = image.to(c.device)  # single scale
            with torch.no_grad():
                outputs = encoder(image)  # (bs, 196, 768)
            
            embeddings = []
            for feature in outputs:
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature))
            embedding = embedding_concat(embeddings[0], embeddings[1])
        
            B, C, H, W = embedding.size()
            S = H*W
            E = B*S    
            
            # (32, 128, h, w)
            p = positionalencoding2d(c.condition_vec, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
            # (32*h*w, 128)
            c_r = p.reshape(B, c.condition_vec, S).transpose(1, 2).reshape(E, c.condition_vec)  # BHWxP
            # (32, h*w, 768)
            e_r = embedding.reshape(B, C, S).transpose(1, 2).reshape(B, H*W, C)  # BHWxC
            e_r = e_r.contiguous()
           
            p_r = generate_prototype(e_r, label)

            e_r = e_r.reshape(-1, C)
            p_r = p_r.reshape(-1, C)
            
            e_pair = e_r - p_r
            # external decoder
            e_loss = train_external_decoder(c, decoder, e_pair, c_r)

            loss = e_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_count += 1

        mean_train_loss = train_loss / train_count
        if c.verbose:
            print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_train_loss, lr))


def test_meta_epoch(c, epoch, loader, encoder, decoder, class_name):
    coreset = coreset_bank[class_name]
    index = faiss.IndexFlatL2(coreset.shape[1])
    index.add(coreset)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index = (index, coreset)

    if c.verbose:
        print('\nCompute loss and scores on category: {}'.format(class_name))
    
    P = c.condition_vec  # 128
    decoder = decoder.eval()
    
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    e_logp_list = []
    test_loss = 0.0
    test_count = 0
    start = time.time()
    with torch.no_grad():
        for n_iter, (image, _, label, mask, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # image: (32, 3, 256), label: (32, ), mask: (32, 1, 256, 256)
            if c.viz:
                image_list.extend(image.numpy())
            gt_label_list.extend(label.numpy())
            gt_mask_list.extend(mask.numpy())
            # data
            image = image.to(c.device) 
            outputs = encoder(image)  
            
            embeddings = []
            for feature in outputs:
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature))
            embedding = embedding_concat(embeddings[0], embeddings[1])
            
            B, C, H, W = embedding.size()
            S = H*W
            E = B*S
           
            # (32, 128, h, w)
            p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
            # (32*h*w, 128)
            c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
            # (32*h*w, 512)
            e_r = embedding.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
            e_r = e_r.contiguous()
            
            # test external decoder
            external_logp, e_loss = test_external_decoder(c, decoder, index, e_r, c_r, size=H)

            loss = e_loss
            #log_prob = (local_logp + global_logp + external_logp) / 3
            test_loss += loss.item()
            test_count += 1
            e_logp_list.append(external_logp)
    
    fps = len(loader.dataset) / (time.time() - start)
    mean_test_loss = test_loss / test_count
    if c.verbose:
        print('Epoch: {:d} \t test_loss: {:.4f} and {:.2f} fps'.format(epoch, mean_test_loss, fps))

    logp = torch.stack(e_logp_list, dim=0)  # (N, 14, 14)
    logp-= torch.max(logp) # normalize likelihoods to (-Inf:0] by subtracting a constant
    prob = torch.exp(logp) # convert to probs in range [0:1]
    
    # upsample
    prob = F.interpolate(prob.unsqueeze(1),
        size=c.crp_size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score maps
    e_score_map = prob.max() - prob # /score_mask.max()  # normality score to anomaly score
    
    score_map = e_score_map 

    return image_list, gt_label_list, gt_mask_list, score_map


def train(c):
    encoder = timm.create_model('tf_efficientnet_b6', features_only=True, 
                out_indices=(2, 3), pretrained=True)
    encoder = encoder.to(c.device).eval()
    pool_dims = encoder.feature_info.channels()

    # Normflow decoder
    decoder = load_decoder_arch(c, sum(pool_dims), c.condition_vec)
    decoder = decoder.to(c.device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=c.lr)
    
    # dataset
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}
    
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = datasets.ImageFolder(os.path.join('data/Mvtec-ImageNet', 'train'), transform=transform_train)
    train_dataset.classes = SEEN_CLASSES
    LABELS = []
    for class_name in SEEN_CLASSES:
        LABELS.append(train_dataset.class_to_idx[class_name])
    samples = []
    for sample in train_dataset.samples:
        label = sample[1]
        if label in LABELS:
            samples.append(sample)
    train_dataset.samples = samples
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=True, **kwargs)

    # stats
    best_img_aucs, best_pixel_aucs = [0]*len(UNSEEN_CLASSES), [0]*len(UNSEEN_CLASSES)
    for epoch in range(c.meta_epochs):
        if c.viz:
            if c.checkpoint:
                load_weights(encoder, decoder, c.checkpoint)
        else:
            print('Train meta epoch: {}'.format(epoch))
            train_meta_epoch(c, epoch, train_loader, encoder, decoder, optimizer)
        
        img_aucs, pixel_aucs = [], []
        best_mean_img_auc, best_mean_pixel_auc = 0, 0
        for class_name in UNSEEN_CLASSES:
            test_dataset = MVTEC('data/mvtec_anomaly_detection', class_name=class_name, train=False,
                norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225], img_size=224, crp_size=224)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)
    
            _, gt_label_list, gt_mask_list, score_map = test_meta_epoch(
                c, epoch, test_loader, encoder, decoder, class_name)

            # calculate detection AUROC
            score_label = np.max(score_map, axis=(1, 2))
            gt_label = np.asarray(gt_label_list, dtype=np.bool)
            img_auc = roc_auc_score(gt_label, score_label)
            
            # calculate segmentation AUROC
            gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
            pixel_auc = roc_auc_score(gt_mask.flatten(), score_map.flatten())

            img_aucs.append(img_auc)
            pixel_aucs.append(pixel_auc)

        for idx, class_name in enumerate(UNSEEN_CLASSES):
            if img_aucs[idx] > best_img_aucs[idx]:
                best_img_aucs[idx] = img_aucs[idx]
            if pixel_aucs[idx] > best_pixel_aucs[idx]:
                best_pixel_aucs[idx] = pixel_aucs[idx]
        if np.mean(img_aucs) > best_mean_img_auc:
            best_mean_img_auc = np.mean(img_aucs)
            state = {'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict()}
            path = os.path.join('weights/protoflow', 'weights_multi_class.pt')
            torch.save(state, path)

        if np.mean(pixel_aucs) > best_mean_pixel_auc:
            best_mean_pixel_auc = np.mean(pixel_aucs) 

    return best_img_aucs, best_pixel_aucs, best_mean_img_auc, best_mean_pixel_auc


def get_args():
    parser = argparse.ArgumentParser(description='ProtoFlow')
    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name: mvtec/stc (default: mvtec)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='file with saved checkpoint')
    parser.add_argument('-cl', '--class-name', default='none', type=str, metavar='C',
                        help='class name for MVTec/STC (default: none)')
    parser.add_argument('-enc', '--enc-arch', default='resnet18', type=str, metavar='A',
                        help='feature extractor')
    parser.add_argument('-dec', '--dec-arch', default='freia-cflow', type=str, metavar='A',
                        help='normalizing flow model (default: freia-cflow)')
    parser.add_argument('-pl', '--pool-layers', default=3, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    parser.add_argument('-cb', '--coupling-blocks', default=4, type=int, metavar='L',
                        help='number of layers used in NF model (default: 8)')
    parser.add_argument('-run', '--run-name', default=0, type=int, metavar='C',
                        help='name of the run (default: 0)')
    parser.add_argument('-inp', '--input-size', default=224, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument("--action-type", default='norm-train', type=str, metavar='T',
                        help='norm-train (default: norm-train)')
    parser.add_argument('-bs', '--batch-size', default=32, type=int, metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--meta-epochs', type=int, default=25, metavar='N',
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub-epochs', type=int, default=8, metavar='N',
                        help='number of sub epochs to train (default: 8)')
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='saves test data visualizations')
    parser.add_argument('--workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default='1', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    
    return args


def main():
    c = get_args()
    warnings.filterwarnings('ignore')
    
    c.img_size = (c.input_size, c.input_size)  # HxW format
    c.crp_size = (c.input_size, c.input_size)  # HxW format
    c.norm_mean, c.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    c.img_dims = [3] + list(c.img_size)
    
    # network hyperparameters
    c.clamp_alpha = 1.9  # see paper equation 2 for explanation
    c.condition_vec = 128
    c.dropout = 0.0  # dropout in s-t-networks
    
    # output settings
    c.verbose = True
    c.hide_tqdm_bar = False
    c.save_results = False
    
    # unsup-train lr settings
    c.print_freq = 2
    c.temp = 0.5
    c.lr_decay_epochs = [i*c.meta_epochs//100 for i in [50,75,90]]
    print('LR schedule: {}'.format(c.lr_decay_epochs))
    c.lr_decay_rate = 0.1
    c.lr_warm_epochs = 2
    c.lr_warm = True
    c.lr_cosine = True
    if c.lr_warm:
        c.lr_warmup_from = c.lr/10.0
        if c.lr_cosine:
            eta_min = c.lr * (c.lr_decay_rate ** 3)
            c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                    1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)) / 2
        else:
            c.lr_warmup_to = c.lr
    
    # setting cuda 
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    c.device = torch.device("cuda" if c.use_cuda else "cpu")
    
    train(c)


if __name__ == '__main__':
    main()