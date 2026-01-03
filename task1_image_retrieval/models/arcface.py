import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # scale
        self.m = m  # margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        # Normalize features and weights
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)

        if labels is not None:
            # Training mode: apply angular margin
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            target_logits = torch.cos(theta + self.m)

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

            output = (one_hot * target_logits) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            # Inference mode: just return cosine similarity
            output = cosine * self.s

        return output


class TankClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_size=512,
        backbone="efficientnet_b3",
        pretrained=True,
    ):
        super().__init__()

        # Backbone - use pretrained model
        if "efficientnet" in backbone:
            self.backbone = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0
            )
            backbone_dim = self.backbone.num_features
        else:
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # type: ignore

        # Projection to embedding space
        self.embedding = nn.Sequential(
            nn.Linear(backbone_dim, embedding_size),  # type: ignore
            nn.BatchNorm1d(embedding_size),
        )

        # ArcFace head
        self.arcface = ArcFaceHead(embedding_size, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = self.embedding(features)

        if labels is not None:
            # Training
            return self.arcface(embeddings, labels), embeddings
        else:
            # Inference - return normalized embeddings
            return F.normalize(embeddings, dim=1)


# Initialize model
tank_types = [
    "M1_Abrams",
    "T72",
    "T90",
    "Leopard_2",
    "Challenger_2",
    "Type_99",
    "Merkava",
]
model = TankClassifier(num_classes=len(tank_types))
