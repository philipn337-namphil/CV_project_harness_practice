"""
model.py
VideoMAE-base(MCG-NJU/videomae-base-finetuned-kinetics) backbone +
dual output head:
  - clip head  : binary classification (illegal/legal)
  - frame head : 16프레임 각각의 event 여부(0/1) - frame-level event detection

기획서 요구사항 매핑:
  - Video Classification             -> clip_logits
  - Frame-level Event Detection      -> frame_logits
  - Heatmap Visualization(Grad-CAM)  -> gradcam.py에서 backbone의 patch embedding
                                         gradient를 이용해 별도로 처리 (이 파일은 forward만 담당)
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel

VIDEOMAE_CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"


class FocalLoss(nn.Module):
    """클래스 불균형(legal:illegal ≈ 1:3 또는 그 반대) 대응용 Focal Loss. gamma=2.0 기획서 고정값."""

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 클래스별 가중치 tensor([w0, w1]) 또는 None
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, num_classes), targets: (B,)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        targets_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)

        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * targets_one_hot * log_probs
        loss = loss.sum(dim=1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GarbageDumpingVideoMAE(nn.Module):
    """
    VideoMAE backbone + dual head.

    forward 입력: pixel_values (B, T=16, C=3, H=224, W=224)
    forward 출력:
      clip_logits  : (B, 2)        - illegal/legal 분류
      frame_logits : (B, T=16)     - 시간축으로 보간한 frame-level event logit
    """

    def __init__(self, pretrained_name=VIDEOMAE_CHECKPOINT, num_clip_classes=2,
                 freeze_backbone_layers=0, dropout=0.1):
        super().__init__()

        self.backbone = VideoMAEModel.from_pretrained(pretrained_name)
        hidden_size = self.backbone.config.hidden_size  # 보통 768

        # 일부 초기 레이어 freeze (선택적, 기본 0 = freeze 없음)
        if freeze_backbone_layers > 0:
            for layer in self.backbone.encoder.layer[:freeze_backbone_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Clip-level head: 전체 패치 평균 풀링 -> binary logits
        self.clip_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_clip_classes),
        )

        # Frame-level head: 패치 토큰을 시간 그룹별로 모아 16개 frame logit 생성
        self.frame_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.num_input_frames = 16

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state  # (B, num_patches_total, hidden_size)
        last_hidden = self.dropout(last_hidden)

        # --- clip head: 전체 패치 평균 풀링 ---
        pooled = last_hidden.mean(dim=1)  # (B, hidden_size)
        clip_logits = self.clip_head(pooled)  # (B, 2)

        # --- frame head: 공간 패치를 평균내 시간 그룹별 표현을 얻고, 16프레임 길이로 보간 ---
        B, N, H = last_hidden.shape
        # VideoMAE patch embedding: tubelet_size=2, spatial 패치 수 = (224/16)^2 = 196
        spatial_patches = (224 // 16) ** 2  # 196
        num_temporal_groups = N // spatial_patches  # 보통 8 (16프레임 / tubelet 2)

        temporal_repr = last_hidden.view(B, num_temporal_groups, spatial_patches, H).mean(dim=2)
        temporal_repr = temporal_repr.transpose(1, 2)  # (B, H, num_temporal_groups)
        temporal_repr = nn.functional.interpolate(
            temporal_repr, size=self.num_input_frames, mode="linear", align_corners=False
        )
        temporal_repr = temporal_repr.transpose(1, 2)  # (B, 16, H)

        frame_logits = self.frame_head(temporal_repr).squeeze(-1)  # (B, 16)

        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "last_hidden_state": last_hidden,  # gradcam.py에서 활용
        }


def build_model(class_counts=None, focal_gamma=2.0, device="cuda"):
    """
    class_counts: {0: legal개수, 1: illegal개수} 형태. 주어지면 Focal Loss alpha를
    inverse frequency로 자동 계산.
    """
    model = GarbageDumpingVideoMAE().to(device)

    alpha = None
    if class_counts:
        total = sum(class_counts.values())
        alpha = torch.tensor([
            total / (2.0 * class_counts.get(0, 1)),
            total / (2.0 * class_counts.get(1, 1)),
        ])

    clip_criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)
    frame_criterion = nn.BCEWithLogitsLoss()  # frame-level은 multi-label 성격이라 BCE 유지

    return model, clip_criterion, frame_criterion


if __name__ == "__main__":
    model, clip_loss_fn, frame_loss_fn = build_model(
        class_counts={0: 384 * 4 + 1152, 1: 1152}, device="cpu"
    )
    dummy = torch.randn(2, 16, 3, 224, 224)
    out = model(dummy)
    print("clip_logits:", out["clip_logits"].shape)
    print("frame_logits:", out["frame_logits"].shape)