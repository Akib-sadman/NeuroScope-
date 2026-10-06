import copy
import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    """
    Swin-Tiny visual backbone (stand-in for "MedCLIP-Swin"). If you have a
    MedCLIP-Swin / MIMIC-EYE-pretrained checkpoint, pass its path via
    ckpt_path and it will be loaded on top of the ImageNet init (strict=False).
    """
    def __init__(self, pretrained=True, ckpt_path=None):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self.embed_dim = self.backbone.num_features  # 768 for swin-tiny

        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            state = state.get("state_dict", state)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"[SwinBackbone] loaded {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")

    def forward_tokens(self, x):
        """Returns patch tokens (B, N, D) before global pooling. N=49 for 224x224 input."""
        feats = self.backbone.forward_features(x)
        if feats.dim() == 4:  # (B, H, W, C)
            b, h, w, c = feats.shape
            feats = feats.reshape(b, h * w, c)
        return feats


class CausalPredictor(nn.Module):
    """8-layer causal Transformer predicting target-token embeddings from context tokens."""
    def __init__(self, dim, depth=8, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=depth)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    @staticmethod
    def _causal_mask(n, device):
        return torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)

    def forward(self, context_tokens, num_targets, pos_embed_slice):
        b, nc, d = context_tokens.shape
        mask_tokens = self.mask_token.expand(b, num_targets, d)
        seq = torch.cat([context_tokens, mask_tokens], dim=1)
        seq = seq + pos_embed_slice[:, : seq.size(1), :]
        attn_mask = self._causal_mask(seq.size(1), seq.device)
        out = self.layers(seq, mask=attn_mask)
        return out[:, nc:, :]


class CrossAttnDecoder(nn.Module):
    """2-layer cross-attention decoder for spatial completion of predicted tokens."""
    def __init__(self, dim, heads=8, depth=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                "norm2": nn.LayerNorm(dim),
            }) for _ in range(depth)
        ])

    def forward(self, queries, context):
        x = queries
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, context, context)
            x = layer["norm1"](x + attn_out)
            x = layer["norm2"](x + layer["mlp"](x))
        return x


class EMA:
    """Exponential moving average copy of a module's parameters (target encoder)."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for sp, p in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


class JEPAModel(nn.Module):
    """
    JEPA-style latent representation prediction:
      context encoder (online Swin) -> causal predictor -> cross-attn decoder,
      trained to match a momentum (EMA) target encoder's representations
      at randomly masked patch positions.
    """
    def __init__(self, pretrained_backbone=True, ckpt_path=None, mask_ratio=0.4, ema_decay=0.999):
        super().__init__()
        self.context_encoder = SwinBackbone(pretrained=pretrained_backbone, ckpt_path=ckpt_path)
        dim = self.context_encoder.embed_dim
        self.target_ema = EMA(self.context_encoder, decay=ema_decay)
        # Register the shadow as a real submodule so calling .to(device) on
        # JEPAModel also moves the EMA target encoder (EMA is a plain
        # Python wrapper, not an nn.Module, so PyTorch won't do this for us).
        self.add_module("_target_shadow", self.target_ema.shadow)
        self.predictor = CausalPredictor(dim, depth=8, heads=8)
        self.decoder = CrossAttnDecoder(dim, heads=8, depth=2)
        self.mask_ratio = mask_ratio

        num_patches = 49  # 7x7 grid for 224-input swin-tiny
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _random_mask(self, n, device):
        num_target = max(1, int(n * self.mask_ratio))
        perm = torch.randperm(n, device=device)
        return perm[num_target:], perm[:num_target]  # context_idx, target_idx

    def forward(self, x):
        tokens = self.context_encoder.forward_tokens(x)
        with torch.no_grad():
            target_tokens = self.target_ema.shadow.forward_tokens(x)

        n = tokens.size(1)
        ctx_idx, tgt_idx = self._random_mask(n, x.device)

        context_tokens = tokens[:, ctx_idx, :] + self.pos_embed[:, ctx_idx, :]
        target_gt = target_tokens[:, tgt_idx, :].detach()

        order = torch.cat([ctx_idx, tgt_idx])
        preds = self.predictor(context_tokens, len(tgt_idx), self.pos_embed[:, order, :])
        preds = self.decoder(preds, context_tokens)

        return nn.functional.smooth_l1_loss(preds, target_gt)

    @torch.no_grad()
    def update_target(self):
        self.target_ema.update(self.context_encoder)


class MRIClassifier(nn.Module):
    """Classification head on top of the (JEPA-pretrained) context encoder."""
    def __init__(self, encoder: SwinBackbone, num_classes=4, freeze_backbone=False):
        super().__init__()
        self.encoder = encoder
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.embed_dim),
            nn.Linear(encoder.embed_dim, num_classes),
        )

    def forward(self, x):
        tokens = self.encoder.forward_tokens(x)
        pooled = tokens.mean(dim=1)
        return self.head(pooled)