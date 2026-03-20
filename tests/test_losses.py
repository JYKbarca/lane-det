import torch
import unittest
import sys
import os

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lane_det.losses import QualityBCELoss, RegLoss

class TestLosses(unittest.TestCase):
    def test_quality_bce_loss(self):
        fl = QualityBCELoss()
        inputs = torch.randn(10, 1)
        targets = torch.rand(10, 1)
        loss = fl(inputs, targets)
        print(f"Quality BCE Loss: {loss.item()}")
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
