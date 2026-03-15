import torch
import torch.nn as nn
from .backbone import ResNet18
from .fpn import LaneFPN
from .head import AnchorHead

class LaneDetector(nn.Module):
    def __init__(self, cfg):
        super(LaneDetector, self).__init__()
        self.cfg = cfg
        
        # Backbone
        self.backbone = ResNet18(pretrained=True)
        # ResNet18 output channels: [64, 128, 256, 512]
        # We use [128, 256, 512] (c3, c4, c5) for FPN usually
        # But LaneFPN uses [c2, c3, c4, c5] by default if configured
        # Let's check config. Default FPN usually takes [c2, c3, c4, c5]
        # Our LaneFPN uses all inputs passed to it.
        # ResNet18 returns [c2, c3, c4, c5]
        
        fpn_out = cfg['model']['fpn_out']
        # Input channels for FPN: [64, 128, 256, 512]
        self.fpn = LaneFPN(
            in_channels_list=[64, 128, 256, 512],
            out_channels=fpn_out,
            use_c2=True # Use C2 for higher resolution
        )
        
        # Head
        model_cfg = cfg.get("model", {})
        self.head = AnchorHead(
            in_channels=fpn_out,
            num_anchors=None, # Dynamic
            use_masked_pooling=model_cfg.get("use_masked_pooling", True),
            debug_shapes=model_cfg.get("debug_shapes", False),
            use_refinement=model_cfg.get("use_refinement", False),
        )

    def forward(self, images, anchors=None, return_aux=False):
        """
        images: [B, 3, H, W]
        anchors: AnchorSet object (optional, if None, must be provided externally or generated)
        """
        # Backbone
        # [c2, c3, c4, c5]
        features = self.backbone(images)
        
        # FPN
        # Returns single high-res feature map [B, C, H/4, W/4]
        fpn_feature = self.fpn(features)
        
        # Head
        if anchors is None:
            raise ValueError("Anchors must be provided to LaneDetector forward")
        
        # Get original image size
        # images: [B, 3, H, W]
        img_h, img_w = images.shape[-2:]
            
        if return_aux:
            cls_logits, reg_preds, aux = self.head(
                fpn_feature, anchors, img_h, img_w, return_aux=True
            )
            return cls_logits, reg_preds, aux

        cls_logits, reg_preds = self.head(fpn_feature, anchors, img_h, img_w, return_aux=False)
        return cls_logits, reg_preds
