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

        depth_backbone = tv_models.resnet50(
            weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        )
        depth_layers = list(depth_backbone.children())[:-1]  # Remove the classification head
        self.depth_backbone = nn.Sequential(*depth_layers)

        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(4096, 5)  # Output: [Rx, Ry, Rz, cx, cy]

    def forward(self, x1, x2, x3):
        feats = self.backbone(x1)
        depth_feats = self.bn(self.relu(self.conv(x2)))
        depth_feats = self.depth_backbone(depth_feats)

        feats = feats.view(feats.size(0), -1)
        depth_feats = depth_feats.view(depth_feats.size(0), -1)

        concat_feats = torch.cat([feats, depth_feats], dim=1)

        return self.fc(concat_feats)


# class SimplePoseNet(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         backbone = tv_models.resnet50(
#             weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
#         )

#         # # Get the original conv1 layer
#         # orig_conv = backbone.conv1

#         # # Create new conv1 layer with 4 input channels
#         # new_conv = nn.Conv2d(
#         #     in_channels=4,
#         #     out_channels=orig_conv.out_channels,
#         #     kernel_size=orig_conv.kernel_size,
#         #     stride=orig_conv.stride,
#         #     padding=orig_conv.padding,
#         #     bias=orig_conv.bias is not None
#         # )
#         # if pretrained:
#         #     with torch.no_grad():
#         #         new_conv.weight[:, :3] = orig_conv.weight  # Copy RGB weights
#         #         # new_conv.weight[:, 3:] = orig_conv.weight[:, :1]  # Init depth like red channel
#         #         # Alternatively: torch.nn.init.kaiming_normal_(new_conv.weight[:, 3:])
                
#         # # Replace the conv1 in the model
#         # backbone.conv1 = new_conv

#         layers = list(backbone.children())[:-1]  # Remove the classification head
#         self.backbone = nn.Sequential(*layers)

#         self.fc1 = nn.Linear(3072, 768)  # Output: [Rx, Ry, Rz, cx, cy]
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm1d(768)
#         self.fc2 = nn.Linear(768, 5)  # Output: [Rx, Ry, Rz, cx, cy]

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.pointnet = SimplePointNetEncoder().to(device)

#     def forward(self, x1, x2, x3):
#         # x = torch.cat([x1, x2], dim=1)
#         feats = self.backbone(x1)
#         point_feats = self.pointnet(x3)
#         feats = feats.view(feats.size(0), -1)
#         total_feats = torch.cat([feats, point_feats], dim=1)
#         total_feats = self.bn(self.relu(self.fc1(total_feats)))
#         return self.fc2(total_feats)




# class SimplePointNetEncoder(nn.Module):
#     def __init__(self, input_dim=3, global_feat=True):
#         super().__init__()
#         self.conv1 = nn.Conv1d(input_dim, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 1024, 1)
#         self.global_feat = global_feat

#     def forward(self, x):
#         # x: (B, N, C)
#         x = x.transpose(2, 1)  # (B, C, N)
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.relu(self.conv2(x))
#         x = self.conv3(x)      # (B, 1024, N)
#         x_max = torch.max(x, 2)[0]  # (B, 1024)

#         return x_max if self.global_feat else x.transpose(2, 1)