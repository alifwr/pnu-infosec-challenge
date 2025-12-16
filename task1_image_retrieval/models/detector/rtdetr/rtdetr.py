import torch.nn as nn 
import torch
import os

from .presnet import PResNet
from .hybrid_encoder import HybridEncoder
from .rtdetrv2_decoder import RTDETRTransformerv2


class RTDETR(nn.Module):
    def __init__(self, num_classes=80, backbone_conf=None, encoder_conf=None, decoder_conf=None, weights=None):
        super().__init__()
        
        # PResNet50 config
        self.backbone_conf = backbone_conf if backbone_conf is not None else {
            'depth': 50, 
            'variant': 'd', 
            'num_stages': 4, 
            'return_idx': [1, 2, 3], 
            'act': 'relu',
            'freeze_norm': False, 
            'pretrained': False
        }
        
        # HybridEncoder config
        self.encoder_conf = encoder_conf if encoder_conf is not None else {
            'in_channels': [512, 1024, 2048],
            'feat_strides': [8, 16, 32],
            'hidden_dim': 256,
            'use_encoder_idx': [2],
            'num_encoder_layers': 1,
            'expansion': 1.0,
            'depth_mult': 1.0
        }
        
        # RTDETRTransformerv2 config
        self.decoder_conf = decoder_conf if decoder_conf is not None else {
            'hidden_dim': 256,
            'feat_channels': [256, 256, 256],
            'feat_strides': [8, 16, 32],
            'num_queries': 300,
            'num_denoising': 100
        }
        
        self.num_classes = num_classes
        self.config = {
            'num_classes': num_classes,
            'backbone': self.backbone_conf,
            'encoder': self.encoder_conf,
            'decoder': self.decoder_conf
        }

        # Submodules
        self.backbone = PResNet(**self.backbone_conf)
        self.encoder = HybridEncoder(**self.encoder_conf)
        self.decoder = RTDETRTransformerv2(num_classes=num_classes, **self.decoder_conf)
        
        if weights is not None:
            self.load_weights(weights)

    def load_weights(self, weights):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights file not found: {weights}")
            
        print(f"Loading weights from {weights}...")
        checkpoint = torch.load(weights, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'ema' in checkpoint and 'module' in checkpoint['ema']:
            state_dict = checkpoint['ema']['module']
        else:
            state_dict = checkpoint
            
        # load_state_dict returns _IncompatibleKeys
        msg = self.load_state_dict(state_dict, strict=False)
        print("Load status:", msg)
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)        
        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 


if __name__ == '__main__':
    import torch
    import os
    
    # Example custom configurations
    custom_backbone_conf = {
        'depth': 34, 
        'freeze_at': -1,
        'freeze_norm': False, 
        'pretrained': True,
        'variant': 'd',
        'return_idx': [1, 2, 3], 
        'num_stages': 4, 
    }

    custom_encoder_conf = {
        'in_channels': [128, 256, 512],
        'hidden_dim': 256,
        'expansion': 0.5,
        'use_encoder_idx': [2],
        'num_encoder_layers': 1,
        'nhead': 8,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'enc_act': 'gelu',
        'expansion': 0.5,
        'depth_mult': 1.0,
        'act': 'silu'
    }

    custom_decoder_conf = {
        'num_layers': 4,
        'feat_channels': [256, 256, 256],
        'feat_strides': [8, 16, 32],
        'hidden_dim': 256,
        'num_levels': 3,
        'num_queries': 300,
        'num_denoising': 100,
        'label_noise_ratio': 0.5,
        'box_noise_scale': 1.0,
        'eval_idx': -1,
        'num_points': [4,4,4],
        'cross_attn_method': 'default',
        'query_select_method': 'default',
        'eval_spatial_size': [640, 640]
    }
    
    # Instantiate model with custom config and weights
    print("Instantiating model with custom backbone, encoder, decoder configs and loading weights...")
    model = RTDETR(
        num_classes=80, 
        backbone_conf=custom_backbone_conf,
        encoder_conf=custom_encoder_conf,
        decoder_conf=custom_decoder_conf,
        weights='weights/rtdetrv2_r34vd_120e_coco_ema.pth'
    )
    model.train()
    
    print("\nModel Configuration:")
    import pprint
    pprint.pprint(model.config)
    
    # Dummy input [B, 3, H, W]
    x = torch.randn(2, 3, 640, 640)
    
    # Dummy targets (required for training with denoising)
    targets = [
        {
            'labels': torch.tensor([1, 2], dtype=torch.long), 
            'boxes': torch.rand(2, 4) # cx, cy, w, h
        } 
        for _ in range(2)
    ]
    
    print("\nRunning forward pass...")
    output = model(x, targets)
    
    print("\nOutput keys:", output.keys())
    if 'pred_logits' in output:
        print("pred_logits:", output['pred_logits'].shape) # [B, num_queries, num_classes]
    if 'pred_boxes' in output:
        print("pred_boxes:", output['pred_boxes'].shape)   # [B, num_queries, 4]