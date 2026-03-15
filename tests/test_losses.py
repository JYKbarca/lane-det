import torch
import unittest
import sys
import os

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lane_det.losses import FocalLoss, RegLoss

class TestLosses(unittest.TestCase):
    def test_focal_loss(self):
        fl = FocalLoss()
        inputs = torch.randn(10, 1)
        targets = torch.randint(0, 2, (10, 1)).float()
        loss = fl(inputs, targets)
        print(f"Focal Loss: {loss.item()}")
        self.assertTrue(loss.item() > 0)

    def test_reg_loss(self):
        rl = RegLoss()
        inputs = torch.randn(10, 4)
        targets = torch.randn(10, 4)
        mask = torch.randint(0, 2, (10, 4)).float()
        loss = rl(inputs, targets, mask)
        print(f"Reg Loss: {loss.item()}")
        self.assertTrue(loss.item() >= 0)

if __name__ == '__main__':
    unittest.main()
