import torch
from torch import nn
from transformers import AutoModel


def initialize_student(config):

    class DINOv3Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(config["student_model"])
            hidden_dim = self.backbone.config.hidden_size

            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, config["embedding_dim"]),
            )

        def forward(self, images):
            features = self.backbone(pixel_values=images).pooler_output
            return self.head(features)

    student = DINOv3Student().to(config["device"])

    backbone_params = sum(p.numel() for p in student.backbone.parameters())
    head_params = sum(p.numel() for p in student.head.parameters())

    print("\nSTUDENT ARCHITECTURE\n")
    print("Backbone:", config["student_model"])
    print(f"Hidden dimension: {student.backbone.config.hidden_size}")
    print(f"Output dimension: {config['embedding_dim']}")
    print(f"Backbone parameters: {backbone_params / 1e6:.2f}M")
    print(f"Head parameters: {head_params / 1e6:.2f}M")
    print(f"Total parameters: {(backbone_params + head_params) / 1e6:.2f}M")
    print("Pretrained backbone: YES")

    return student
