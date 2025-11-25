# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

import common
import metrics
from utils import plot_segmentation_images
import wandb

LOGGER = logging.getLogger(__name__)

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x

class NoiseGenerator(torch.nn.Module):
    def __init__(self, C, Z=64):
        super(NoiseGenerator, self).__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(C+Z,C), torch.nn.Tanh())
        
    def forward(self, feat, z):
        combined_input = torch.cat([feat,z], dim=1)
        return self.model(combined_input)
        

class Projection(torch.nn.Module):
    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1

class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1, 
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        **kwargs,
    ):
        pid = os.getpid()
        def show_mem():
            return(psutil.Process(pid).memory_info())
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin 

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None
        
        self.teacher = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.teacher.to(self.device)
        self.ema_decay = 0.99
        
        self.z_dim=64
        self.noise_generator = NoiseGenerator(C=self.target_embed_dimension, Z=self.z_dim).to(self.device)
        self.g_opt = torch.optim.Adam(self.noise_generator.parameters(), lr=self.dsc_lr * 2.0, weight_decay=1e-5)
        self.grad_clip = 1.0
        
    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)
    

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling        


        return features, patch_shapes

    
    def test(self, training_data, test_data, save_segmentation_images):

        ckpt_path = os.path.join(self.ckpt_dir, "models.ckpt")
        if os.path.exists(ckpt_path):
            state_dicts = torch.load(ckpt_path, map_location=self.device)
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt, anomalies_gt = self.predict(test_data)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)
            
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )["auroc"]

        # Compute PRO score & PW Auroc for all images
        pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]

        return auroc, full_pixel_auroc
    
    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt, anomalies=None):
        
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)
        
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt 
        )["auroc"]
        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)


            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
                # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), 
                                            norm_segmentations)
        else:
            full_pixel_auroc = -1 
            pro = -1

        if anomalies is not None and len(anomalies) > 0:

            print("\n" + "="*70)
            print("Class-wise Performance:")
            print("="*70)
            print(f"{'Class':<15} {'Count':<8} {'I-AUROC':<10} {'P-AUROC':<10} {'PRO':<10}")
            print("-"*70)
            
            # 고유한 클래스 찾기
            unique_classes = []
            for anomaly in anomalies:
                if anomaly not in unique_classes:
                    unique_classes.append(anomaly)
            
            classwise_results = {}
            
            # Normal 데이터 인덱스
            normal_indices = [i for i, a in enumerate(anomalies) if a == "good"]
            
            for class_name in unique_classes:
                # 해당 클래스 인덱스
                class_indices = [i for i, a in enumerate(anomalies) if a == class_name]
                count = len(class_indices)
                
                if class_name == "good":
                    print(f"{class_name:<15} {count:<8} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
                else:
                    # Normal + 해당 Anomaly만 사용
                    combined_indices = normal_indices + class_indices
                    
                    # 데이터 추출
                    class_scores = scores[combined_indices]
                    class_labels = np.array([labels_gt[i] for i in combined_indices])
                    
                    # Image AUROC
                    class_auroc = metrics.compute_imagewise_retrieval_metrics(
                        class_scores, class_labels
                    )["auroc"]
                    
                    # Pixel AUROC & PRO
                    if len(masks_gt) > 0:
                        class_segmentations = segmentations[combined_indices]
                        class_masks = [masks_gt[i] for i in combined_indices]
                        
                        # Normalization
                        min_scores = (
                            class_segmentations.reshape(len(class_segmentations), -1)
                            .min(axis=-1)
                            .reshape(-1, 1, 1, 1)
                        )
                        max_scores = (
                            class_segmentations.reshape(len(class_segmentations), -1)
                            .max(axis=-1)
                            .reshape(-1, 1, 1, 1)
                        )
                        class_norm_segmentations = np.zeros_like(class_segmentations)
                        for min_score, max_score in zip(min_scores, max_scores):
                            class_norm_segmentations += (class_segmentations - min_score) / max(max_score - min_score, 1e-2)
                        class_norm_segmentations = class_norm_segmentations / len(class_scores)
                        
                        class_pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                            class_norm_segmentations, class_masks
                        )
                        class_pixel_auroc = class_pixel_scores["auroc"]
                        class_pro = metrics.compute_pro(
                            np.squeeze(np.array(class_masks)), 
                            class_norm_segmentations
                        )
                    else:
                        class_pixel_auroc = -1
                        class_pro = -1
                    
                    print(f"{class_name:<15} {count:<8} {class_auroc:<10.4f} {class_pixel_auroc:<10.4f} {class_pro:<10.4f}")
                    
                    classwise_results[class_name] = {
                        'image_auroc': class_auroc,
                        'pixel_auroc': class_pixel_auroc,
                        'pro': class_pro,
                        'count': count
                    }
        
            # 평균 (Anomaly classes only)
            if len(classwise_results) > 0:
                avg_auroc = np.mean([r['image_auroc'] for r in classwise_results.values()])
                avg_pixel_auroc = np.mean([r['pixel_auroc'] for r in classwise_results.values() if r['pixel_auroc'] > 0])
                avg_pro = np.mean([r['pro'] for r in classwise_results.values() if r['pro'] > 0])
                
                print("-"*70)
                print(f"{'Avg (Anomaly)':<15} {'':<8} {avg_auroc:<10.4f} {avg_pixel_auroc:<10.4f} {avg_pro:<10.4f}")
            
            print("="*70 + "\n")

        return auroc, full_pixel_auroc, pro
        
    
    def train(self, training_data, test_data):
        
        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            
            
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            self.predict(training_data, "train_")
            scores, segmentations, features, labels_gt, masks_gt, anomalies_gt = self.predict(test_data)
            auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt, anomalies_gt)
            
            return auroc, full_pixel_auroc, anomaly_pixel_auroc
        
        def update_state_dict(d):
            
            state_dict["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})

        best_record = None
        for i_mepoch in range(self.meta_epochs):
            
            if i_mepoch > 0:
                self._train_discriminator(training_data, i_mepoch, teacher=True)
            else:
                self._train_discriminator(training_data, i_mepoch)

            # torch.cuda.empty_cache()
            scores, segmentations, features, labels_gt, masks_gt, anomalies_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(test_data, scores, segmentations, features, labels_gt, masks_gt, anomalies_gt)
            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
            self.logger.logger.add_scalar("pro", pro, i_mepoch)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(state_dict)
                self.teacher.load_state_dict(state_dict['discriminator'])
                # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                    #self.teacher.load_state_dict(state_dict['discriminator'])
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro 
                    update_state_dict(state_dict)                
                    #self.teacher.load_state_dict(state_dict['discriminator'])
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")

            wandb.log({
                "meta_epoch"   : i_mepoch,
                "I-AUROC": auroc,
                "P-AUROC": full_pixel_auroc,
                "PRO-AUROC": pro
            })
        
        torch.save(state_dict, ckpt_path)
        
        return best_record
            

    def _train_discriminator(self, input_data, meta_epoch,  teacher=False):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        self.teacher.eval()
        self.noise_generator.train()
        
        cur_progress = meta_epoch / self.meta_epochs
        warmup_ratio = 0.25
        transition_ratio = 0.75
        
        mix_ratio = 0.0
        if cur_progress >= warmup_ratio and cur_progress < (warmup_ratio + transition_ratio):
            mix_ratio = (cur_progress - warmup_ratio) / transition_ratio * 0.5
        if cur_progress >= (warmup_ratio + transition_ratio):
            mix_ratio = 0.5
        
        
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                    else:
                        true_feats = self._embed(img, evaluation=False)[0]
                    
                    noise_gaussian = torch.zeros_like(true_feats)
                    noise_adv = torch.zeros_like(true_feats)
                    
                    if cur_progress < (warmup_ratio + transition_ratio) :
                        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
                        noise = torch.stack([
                            torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                            for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
                        noise_gaussian = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                        
                    if cur_progress >= warmup_ratio:
                        with torch.no_grad():
                            z = torch.randn(true_feats.shape[0], self.z_dim).to(self.device)
                            noise_adv = self.noise_generator(true_feats, z).detach() * self.noise_std
                    
                    mix_noise = (1 - mix_ratio)*noise_gaussian + mix_ratio * noise_adv
                    fake_feats = true_feats + mix_noise
                    
                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                    
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]
                    
                    th = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)                 

                    loss_gan = true_loss.mean() + fake_loss.mean()
                    
                    if teacher is True:
                        with torch.no_grad():
                            teacher_scores = self.teacher(torch.cat([true_feats, fake_feats])).detach()
                            teacher_true = teacher_scores[:len(true_feats)]
                            teacher_fake = teacher_scores[len(fake_feats):]
                            
                        # teacher_delta = (teacher_true - teacher_fake)
                        # student_delta = (true_scores - fake_scores)
                        
                        # loss_kd = 0.1 * torch.abs(teacher_delta - student_delta).mean()
                        
                        student_logits = torch.cat([true_scores, fake_scores], dim=1)
                        teacher_logits = torch.cat([teacher_true, teacher_fake], dim=1)

                        # 2. KL Divergence Loss 계산
                        loss_kd = F.kl_div(
                            F.log_softmax(student_logits / 4, dim=1),
                            F.softmax(teacher_logits / 4, dim=1),
                            reduction='batchmean'
                        ) * (4 * 4) # T^2 스케일링
                                                
                        
                    else:
                        loss_kd = 0.0
                    
                    loss = loss_gan + loss_kd
                    loss.backward()
                    
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    torch.nn.utils.clip_grad_norm(self.discriminator.parameters(), self.grad_clip)
                    
                    if self.pre_proj > 0:
                        torch.nn.utils.clip_grad_norm(self.pre_projection.parameters(), self.grad_clip)
                        self.proj_opt.step()
                    if self.train_backbone:
                        torch.nn.utils.clip_grad_norm(self.backbone.parameters(), self.grad_clip)
                        self.backbone_opt.step()
                    self.dsc_opt.step()
                    
                    
                    with torch.no_grad():
                        for t_param, s_param in zip(self.teacher.parameters(), self.discriminator.parameters()):
                            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1-self.ema_decay)
                    
                    loss_g = 0.0
                    
                    if cur_progress >= warmup_ratio:
                        self.g_opt.zero_grad()
                        
                        true_feats_detached = true_feats.detach()
                        
                        z = torch.randn(true_feats.shape[0], self.z_dim).to(self.device)
                        noise_adv = self.noise_generator(true_feats_detached, z) * self.noise_std
                        fake_feats = true_feats_detached + noise_adv
                        
                        for p in self.discriminator.parameters():
                            p.requires_grad=False
                            
                        fake_scores = self.discriminator(fake_feats)
                        loss_g = torch.clip(-fake_scores + th, min=0).mean()
                        loss_g.backward()
                        
                        torch.nn.utils.clip_grad_norm(self.noise_generator.parameters(), self.grad_clip)
                        
                        self.g_opt.step()  
                         
                        for p in self.discriminator.parameters():
                            p.requires_grad=True                              

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                    
                    
                    loss_dict = {
                        "loss_gan": loss_gan.item(),
                        "p_true": p_true.item(),
                        "p_fake": p_fake.item(),
                        "loss_kd": loss_kd,
                        "loss_d": loss.item(),
                        "loss_g": loss_g,
                    }
                                     
                    wandb.log(loss_dict, step=self.logger.g_iter)
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                if self.cos_lr:
                    self.dsc_schl.step()
                
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    # def _predict_dataloader(self, dataloader, prefix):
    #     """This function provides anomaly scores/maps for full dataloaders."""
    #     _ = self.forward_modules.eval()


    #     img_paths = []
    #     scores = []
    #     masks = []
    #     features = []
    #     labels_gt = []
    #     masks_gt = []
    #     from sklearn.manifold import TSNE

    #     with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
    #         for data in data_iterator:
    #             if isinstance(data, dict):
    #                 labels_gt.extend(data["is_anomaly"].numpy().tolist())
    #                 if data.get("mask", None) is not None:
    #                     masks_gt.extend(data["mask"].numpy().tolist())
    #                 image = data["image"]
    #                 img_paths.extend(data['image_path'])
    #             _scores, _masks, _feats = self._predict(image)
    #             for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
    #                 scores.append(score)
    #                 masks.append(mask)

    #     return scores, masks, features, labels_gt, masks_gt

    def _predict_dataloader(self, dataloader, prefix):
        """
        [수정됨]
        전체 데이터로더를 순회하며, 4D(Train) 또는 5D(Test) 텐서를 처리하고
        결과를 Python list로 집계(aggregate)합니다.
        """
        _ = self.forward_modules.eval()

        # 최종 반환을 위한 빈 리스트 초기화
        all_scores = []
        all_masks = []
        all_features = []
        all_labels_gt = []
        all_masks_gt = []
        all_anomalies = []

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    # Ground truth 정보 추출
                    labels_gt = data["is_anomaly"].numpy().tolist()
                    # if data.get("mask", None) is not None:
                    #     all_masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    anomalies = data['anomaly']
                    # img_paths.extend(data['image_path']) # (필요시 주석 해제)
                else:
                    # 데이터가 딕셔너리가 아닌 경우 (예: 단순 텐서)
                    image = data
                    labels_gt = []
                    all_masks_gt = []
                    anomalies = []

                # --- 텐서 차원 분기 (4D vs 5D) ---
                
                if image.dim() == 5:
                    # --- [ETRI TEST 경로] ---
                    # image.shape == [B, P, C, H, W] (예: [8, 9, 3, 224, 224])
                    B, P, C, H, W = image.shape
                    
                    # 1. 5D 텐서를 4D로 Unroll (Flatten)
                    # [B, P, C, H, W] -> [B*P, C, H, W] (예: [72, 3, 224, 224])
                    image = image.view(-1, C, H, W)
                    
                    # 2. _predict() 호출
                    # _predict는 [72, ...] 크기의 Python list들을 반환합니다.
                    _scores_patches_list, _masks_patches_list, _feats_patches_list = self._predict(image)

                    # [FIX] 3. list를 Tensor로 변환 (오류가 발생했던 지점)
                    # (device를 명시하여 GPU에서 집계 연산 수행)
                    _scores_patches_tensor = torch.tensor(np.array(_scores_patches_list), device=self.device)
                    _masks_patches_tensor = torch.tensor(np.array(_masks_patches_list), device=self.device)
                    _feats_patches_tensor = torch.tensor(np.array(_feats_patches_list), device=self.device)

                    # 4. I-AUROC Roll-up (Aggregation)
                    # [72] -> [B, P] (예: [8, 9])
                    _scores_patches_tensor = _scores_patches_tensor.view(B, P)
                    # "9개 중 하나라도 wrong이면" -> max() 연산
                    # [B, P] -> [B] (예: [8])
                    _scores_tensor, _ = torch.max(_scores_patches_tensor, dim=1)
                    
                    # 5. P-AUROC(마스크) Roll-up
                    # [72, H, W] -> [B, P, H, W]
                    _masks_tensor = _masks_patches_tensor.view(B, P, H, W)
                    # [B, P, H, W] -> [B, H, W] (9개 패치 중 max score)
                    _masks_tensor, _ = torch.max(_masks_tensor, dim=1)
                    
                    # 6. 피처 Roll-up (첫 번째 패치만 대표로 저장)
                    _feats_tensor = _feats_patches_tensor.view(B, P, -1)
                    _feats_tensor = _feats_tensor[:, 0, :] # [B, C_feat]
                    
                    # 7. 최종 집계 리스트에 추가하기 위해 CPU/list로 변환
                    _scores = _scores_tensor.cpu().numpy().tolist()
                    _masks = _masks_tensor.cpu().numpy().tolist()
                    _feats = _feats_tensor.cpu().numpy().tolist()

                else:
                    # --- [MVTec / ETRI TRAIN 경로] ---
                    # image.shape == [B, C, H, W] (예: [64, 3, 224, 224])
                    # _predict는 [B] 크기의 list들을 반환
                    _scores, _masks, _feats = self._predict(image)

                # --- [공통] 집계 ---
                # _scores, _masks, _feats는 4D/5D 경로 모두에서 Python list 상태임
                all_scores.extend(_scores)
                #all_masks.extend(_masks)
                #all_features.extend(_feats)
                all_labels_gt.extend(labels_gt if isinstance(labels_gt, list) else [labels_gt])
                all_anomalies.extend(anomalies if isinstance(anomalies, list) else [anomalies])

        # _predict_dataloader의 최종 반환 (list 형태)
        return all_scores, all_masks, all_features, all_labels_gt, all_masks_gt, all_anomalies

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True, 
                                                 evaluation=True)
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
