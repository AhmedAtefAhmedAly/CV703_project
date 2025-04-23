# bpc/pose/models/simple_pose_net.py

import torch
import torch.nn as nn
import torchvision.models as tv_models

# class SimplePoseNet(nn.Module):
#     def __init__(self, loss_type="euler", pretrained=True):
#         """
#         Args:
#             loss_type: one of "euler", "quat", or "6d". Determines the number of output neurons.
#             pretrained: if True, use pretrained ResNet50 weights.
#         """
#         super(SimplePoseNet, self).__init__()
#         # Load a ResNet50 backbone.
#         backbone = tv_models.resnet50(
#             weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
#         )
#         layers = list(backbone.children())[:-1]  # Remove the classification head.
#         self.backbone = nn.Sequential(*layers)
        
#         # Determine rotation output dimension.
#         if loss_type == "euler":
#             out_dim = 3
#         elif loss_type == "quat":
#             out_dim = 4
#         elif loss_type == "6d":
#             out_dim = 6
#         else:
#             raise ValueError("loss_type must be one of 'euler', 'quat', or '6d'")
        
#         self.fc = nn.Linear(2048, out_dim)  # Only rotation outputs.

#     def forward(self, x):
#         feats = self.backbone(x)
#         feats = feats.view(feats.size(0), -1)
#         preds = self.fc(feats)
#         return preds
    
import torch
import torch.nn as nn
import torchvision.models as tv_models

class SimplePoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = tv_models.resnet50(
            weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        )
        layers = list(backbone.children())[:-1]  # Remove the classification head
        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(2048, 5)  # Output: [Rx, Ry, Rz, cx, cy]

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        return self.fc(feats)
    
import torch
import torch.nn as nn
import torchvision.models as tv_models

class MultiModalPoseNetLate(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # RGB branch using a ResNet50 backbone.
        rgb_backbone = tv_models.resnet50(
            weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        )
        # Remove the classification head.
        self.rgb_branch = nn.Sequential(*list(rgb_backbone.children())[:-1])  # Output: [B, 2048, 1, 1]
        
        # Depth branch: a small CNN tailored for single-channel depth input.
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 64, 1, 1]
        )
        
        # Polarization branch: similar to depth branch.
        self.pol_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: [B, 64, 1, 1]
        )
        
        # Fusion: concatenating features from the three branches.
        # Total feature dimension: 2048 (RGB) + 64 (depth) + 64 (pol) = 2176.
        self.fc = nn.Linear(2048 + 64 + 64, 5)  # For pose: [Rx, Ry, Rz, cx, cy]

    def forward(self, rgb, depth, polar):
        # Extract features from each branch.
        rgb_feat = self.rgb_branch(rgb)  # [B, 2048, 1, 1]
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)  # [B, 2048]
        
        depth_feat = self.depth_branch(depth)  # [B, 64, 1, 1]
        depth_feat = depth_feat.view(depth_feat.size(0), -1)  # [B, 64]
        
        pol_feat = self.pol_branch(polar)  # [B, 64, 1, 1]
        pol_feat = pol_feat.view(pol_feat.size(0), -1)  # [B, 64]
        
        # Fuse features by concatenation.
        fused = torch.cat([rgb_feat, depth_feat, pol_feat], dim=1)  # [B, 2176]
        preds = self.fc(fused)
        return preds
