import torch
import unittest
import sys
import os
import yaml
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lane_det.models import LaneDetector
from lane_det.anchors.anchor_generator import AnchorGenerator

class TestModels(unittest.TestCase):
    def setUp(self):
        # Mock config
        self.cfg = {
            'dataset': {
                'img_size': [1280, 720],
                'y_samples': 56
            },
            'anchor': {
                'x_positions': 10,
                'angles': [-20, 0, 20],
                'num_y': 56
            },
            'model': {
                'fpn_out': 64,
                'head_channels': 64
            }
        }
        
        # Generate anchors
        self.anchor_generator = AnchorGenerator.from_config(self.cfg, cache_dir=None)
        self.anchors = self.anchor_generator.generate()

    def test_detector_forward(self):
        model = LaneDetector(self.cfg)
        model.eval()
        
        # Create dummy input [B, 3, H, W]
        B = 2
        H, W = 720, 1280
        images = torch.randn(B, 3, H, W)
        
        # Forward
        cls_logits, reg_preds = model(images, self.anchors)
        
        # Check shapes
        num_anchors = len(self.anchors.anchor_xs)
        num_y = len(self.anchors.y_samples)
        
        print(f"Cls shape: {cls_logits.shape}")
        print(f"Reg shape: {reg_preds.shape}")
        
        self.assertEqual(cls_logits.shape, (B, num_anchors))
        self.assertEqual(reg_preds.shape, (B, num_anchors, num_y))
        
        # Check values range (roughly)
        # Cls logits can be anything
        # Reg preds can be anything
        
if __name__ == '__main__':
    unittest.main()
