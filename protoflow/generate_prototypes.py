import os
import timm
import argparse
import numpy as np
from tqdm import tqdm

import torch
import faiss
from torch.nn import functional as F
from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import DataLoader
from datasets.mvtec_test import MVTEC
from sampling_methods.kcenter_greedy import kCenterGreedy


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


class PrototypeModule(object):
    def __init__(self, args):
        super(PrototypeModule, self).__init__()

        self.args = args

        self.model = timm.create_model('tf_efficientnet_b6', features_only=True, 
                out_indices=(2, 3), pretrained=True)
        self.model.to(args.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        train_dataset = MVTEC(args.data_path, class_name=args.class_name, train=True,
                    norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225], img_size=224, crp_size=224)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        self.embedding_path = os.path.join(f'weights/prototypes', self.args.class_name)
        os.makedirs(self.embedding_path, exist_ok=True)     

    def run(self):
        if os.path.exists(os.path.join(self.embedding_path, 'index.faiss')):
            return
        else:
            self.embedding_list = []
            for batch in tqdm(self.train_loader):
                x, _, _, file_name, _ = batch
                features = self.model(x.to(self.args.device))
                embeddings = []
                for feature in features:
                    m = torch.nn.AvgPool2d(3, 1, 1)
                    # embeddings.append(m(feature))
                    feature = m(feature)
                    embeddings.append(feature)
                embedding = embedding_concat(embeddings[0], embeddings[1])
                self.embedding_list.extend(reshape_embedding(embedding.cpu().numpy()))

            total_embeddings = np.array(self.embedding_list)
            print(total_embeddings.shape)
            # Random projection
            self.randomprojector = SparseRandomProjection(n_components=272, eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
            self.randomprojector.fit(total_embeddings)
            # Coreset Subsampling
            selector = kCenterGreedy(total_embeddings, 0, 0)
            selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*self.args.coreset_sampling_ratio))
            self.embedding_coreset = total_embeddings[selected_idx]
            
            print('initial embedding size : ', total_embeddings.shape)
            print('final embedding size : ', self.embedding_coreset.shape)
            # faiss
            self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
            self.index.add(self.embedding_coreset) 
            np.save(os.path.join(self.embedding_path, 'coreset.npy'), self.embedding_coreset)
            faiss.write_index(self.index,  os.path.join(self.embedding_path, 'index.faiss'))


def get_args():
    parser = argparse.ArgumentParser(description='Prototypes Generation')
    parser.add_argument('--data_path', default='/disk/yxc/datasets/mvtec_anomaly_detection') 
    parser.add_argument('--class_name', default='bottle')
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--coreset_sampling_ratio', default=0.01)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for class_name in MVTEC_CLASS_NAMES:
        args.class_name = class_name
        model = PrototypeModule(args=args)
        model.run()
