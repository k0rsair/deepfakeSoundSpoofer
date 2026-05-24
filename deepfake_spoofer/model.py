from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: Tensor) -> Tensor:
        values_per_channel = input.numel() // input.size(1)
        if self.training and values_per_channel <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class SafeBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: Tensor) -> Tensor:
        values_per_channel = input.numel() // input.size(1)
        if self.training and values_per_channel <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class Wav2VecFrontend(nn.Module):
    def __init__(
        self,
        bundle_name: str = "WAV2VEC2_XLSR_300M",
        *,
        freeze_wav2vec: bool = False,
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 0,
        wav2vec_layers: int | None = None,
    ) -> None:
        super().__init__()
        import torchaudio

        if not hasattr(torchaudio.pipelines, bundle_name):
            raise ValueError(f"torchaudio.pipelines has no bundle named {bundle_name!r}")

        self.bundle_name = bundle_name
        self.bundle = getattr(torchaudio.pipelines, bundle_name)
        self.wav2vec = self.bundle.get_model()
        self.sample_rate = int(self.bundle.sample_rate)
        self.wav2vec_layers = wav2vec_layers if wav2vec_layers and wav2vec_layers > 0 else None
        self.freeze_wav2vec = freeze_wav2vec

        params: dict[str, Any] = getattr(self.bundle, "_params", {})
        self.out_dim = int(params.get("encoder_embed_dim", 1024))

        if freeze_wav2vec:
            for parameter in self.wav2vec.parameters():
                parameter.requires_grad = False
        else:
            if freeze_feature_extractor and hasattr(self.wav2vec, "feature_extractor"):
                for parameter in self.wav2vec.feature_extractor.parameters():
                    parameter.requires_grad = False
            self._freeze_first_transformer_layers(freeze_transformer_layers)

    def _freeze_first_transformer_layers(self, count: int) -> None:
        if count <= 0:
            return
        transformer = getattr(getattr(self.wav2vec, "encoder", None), "transformer", None)
        layers = getattr(transformer, "layers", None)
        if layers is None:
            return
        for layer in layers[:count]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, waveforms: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if self.freeze_wav2vec:
            self.wav2vec.eval()
            with torch.no_grad():
                features, feature_lengths = self.wav2vec.extract_features(
                    waveforms,
                    lengths,
                    num_layers=self.wav2vec_layers,
                )
        else:
            features, feature_lengths = self.wav2vec.extract_features(
                waveforms,
                lengths,
                num_layers=self.wav2vec_layers,
            )

        return features[-1], feature_lengths


def lengths_to_mask(lengths: Tensor, max_length: int, *, device: torch.device | None = None) -> Tensor:
    if device is None:
        device = lengths.device
    safe_lengths = lengths.to(device=device, dtype=torch.long).clamp_min(1).clamp_max(max_length)
    positions = torch.arange(max_length, device=device).unsqueeze(0)
    return positions < safe_lengths.unsqueeze(1)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.norm = nn.GroupNorm(1, dim)
        self.pointwise1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.pointwise2 = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        residual = x
        out = self.depthwise(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.pointwise1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.pointwise2(out)
        out = self.drop(out)
        out = out + residual
        if mask is not None:
            out = out * mask.unsqueeze(1).to(dtype=out.dtype)
        return out


class AttentiveStatisticsPool(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int | None = None, eps: float = 1e-5) -> None:
        super().__init__()
        attention_dim = hidden_dim or input_dim
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.eps = eps

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp_min(1).clamp_max(x.size(1))

        mask = lengths_to_mask(lengths, x.size(1), device=x.device)
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        mean = torch.sum(weights * x, dim=1)
        second_moment = torch.sum(weights * x.square(), dim=1)
        std = (second_moment - mean.square()).clamp_min(self.eps).sqrt()
        return torch.cat([mean, std], dim=1)


class Wav2VecTemporalHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        embedding_dim: int = 160,
        num_blocks: int = 4,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TemporalConvBlock(
                model_dim,
                kernel_size=5,
                dilation=2 ** (block_index % 3),
                dropout=dropout,
            )
            for block_index in range(num_blocks)
        )
        self.pool = AttentiveStatisticsPool(model_dim, hidden_dim=model_dim)
        self.embedding_proj = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(model_dim * 2, embedding_dim),
            nn.GELU(),
        )
        self.output_drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if lengths is None:
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.long, device=features.device)
        else:
            lengths = lengths.to(device=features.device, dtype=torch.long).clamp_min(1).clamp_max(features.size(1))

        mask = lengths_to_mask(lengths, features.size(1), device=features.device)
        x = self.input_proj(features)
        x = self.input_norm(x)
        x = F.gelu(x)
        x = self.input_drop(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x, mask)
        x = x.transpose(1, 2)
        pooled = self.pool(x, lengths)
        embedding = self.embedding_proj(pooled)
        logits = self.classifier(self.output_drop(embedding))
        return embedding, logits


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, temperature: float = 1.0) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = SafeBatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temperature = temperature
        nn.init.xavier_normal_(self.att_weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)
        original_size = x.size()
        x = self.bn(x.reshape(-1, original_size[-1])).reshape(original_size)
        return self.act(x)

    def _derive_att_map(self, x: Tensor) -> Tensor:
        node_count = x.size(1)
        left = x.unsqueeze(2).expand(-1, -1, node_count, -1)
        right = left.transpose(1, 2)
        att_map = torch.tanh(self.att_proj(left * right))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temperature
        return F.softmax(att_map, dim=-2)


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_proj_master = nn.Linear(in_dim, out_dim)
        self.att_weight11 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight_master = nn.Parameter(torch.empty(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_att_master = nn.Linear(in_dim, out_dim)
        self.proj_without_att_master = nn.Linear(in_dim, out_dim)
        self.bn = SafeBatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temperature = temperature

        for parameter in (
            self.att_weight11,
            self.att_weight22,
            self.att_weight12,
            self.att_weight_master,
        ):
            nn.init.xavier_normal_(parameter)

    def forward(self, x1: Tensor, x2: Tensor, master: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)

        original_size = x.size()
        x = self.bn(x.reshape(-1, original_size[-1])).reshape(original_size)
        x = self.act(x)
        return x.narrow(1, 0, num_type1), x.narrow(1, num_type1, num_type2), master

    def _derive_att_map(self, x: Tensor, num_type1: int, num_type2: int) -> Tensor:
        node_count = x.size(1)
        left = x.unsqueeze(2).expand(-1, -1, node_count, -1)
        right = left.transpose(1, 2)
        att_map = torch.tanh(self.att_proj(left * right))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :],
            self.att_weight11,
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :],
            self.att_weight22,
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :],
            self.att_weight12,
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :],
            self.att_weight12,
        )
        att_board = att_board / self.temperature
        return F.softmax(att_board, dim=-2)

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_proj_master(att_map))
        att_map = torch.matmul(att_map, self.att_weight_master)
        att_map = att_map / self.temperature
        att_map = F.softmax(att_map, dim=-2)
        return self.proj_with_att_master(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x)) + (
            self.proj_without_att_master(master)
        )


class GraphPool(nn.Module):
    def __init__(self, keep_ratio: float, in_dim: int, dropout: float) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor) -> Tensor:
        scores = self.sigmoid(self.proj(self.drop(h)))
        return self._top_k_graph(scores, h)

    def _top_k_graph(self, scores: Tensor, h: Tensor) -> Tensor:
        _, node_count, feature_count = h.size()
        kept_nodes = max(int(node_count * self.keep_ratio), 1)
        _, index = torch.topk(scores, kept_nodes, dim=1)
        index = index.expand(-1, -1, feature_count)
        return torch.gather(h * scores, 1, index)


class ResidualBlock(nn.Module):
    def __init__(self, channels: list[int], *, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = SafeBatchNorm2d(num_features=channels[0])
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = SafeBatchNorm2d(num_features=channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(2, 3), padding=(0, 1), stride=1)
        self.downsample = channels[0] != channels[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(channels[0], channels[1], kernel_size=(1, 3), padding=(0, 1), stride=1)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.conv_downsample(x) if self.downsample else x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.conv1(out)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


class MFCCResNetBlock(nn.Module):
    def __init__(self, in_depth: int, depth: int, *, first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.pre_bn = SafeBatchNorm2d(in_depth)
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = SafeBatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)

    def forward(self, signal: Tensor) -> Tensor:
        residual = self.conv11(signal)
        out = signal if self.first else self.lrelu(self.pre_bn(signal))
        out = self.conv1(out)
        out = self.lrelu(self.bn1(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + residual


def build_mfcc_transform(
    *,
    sample_rate: int = 16_000,
    n_mfcc: int = 40,
    n_mels: int = 64,
) -> nn.Module:
    import torchaudio

    return torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": int(0.025 * sample_rate),
            "hop_length": int(0.010 * sample_rate),
            "n_mels": n_mels,
            "center": True,
            "power": 2.0,
        },
    )


def normalize_mfcc_features(mfcc: Tensor) -> Tensor:
    mean = mfcc.mean(dim=(-1, -2), keepdim=True)
    std = mfcc.std(dim=(-1, -2), keepdim=True).clamp_min(1e-5)
    return (mfcc - mean) / std


class MFCCResNetBranch(nn.Module):
    """PyAra MFCC/ResNet branch that exposes the 128-d embedding before fc2."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        n_mfcc: int = 40,
        n_mels: int = 64,
        embedding_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.mfcc = build_mfcc_transform(sample_rate=sample_rate, n_mfcc=n_mfcc, n_mels=n_mels)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = MFCCResNetBlock(32, 32, first=True)
        self.block2 = MFCCResNetBlock(32, 32)
        self.block3 = MFCCResNetBlock(32, 32)
        self.block4 = MFCCResNetBlock(32, 32)
        self.block5 = MFCCResNetBlock(32, 32)
        self.block6 = MFCCResNetBlock(32, 32)
        self.block7 = MFCCResNetBlock(32, 32)
        self.block8 = MFCCResNetBlock(32, 32)
        self.block9 = MFCCResNetBlock(32, 32)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.bn = SafeBatchNorm2d(32)
        self.lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        waveforms: Tensor | None,
        lengths: Tensor | None = None,
        *,
        mfcc_features: Tensor | None = None,
        mfcc_feature_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del lengths, mfcc_feature_lengths
        if mfcc_features is None:
            if waveforms is None:
                raise ValueError("waveforms are required when mfcc_features are not provided")
            x = self._extract_mfcc(waveforms)
        else:
            x = self._prepare_cached_mfcc(mfcc_features)
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.mp(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.mp(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.lrelu(self.bn(x))
        x = self.mp(x)
        x = self.pool(x).flatten(1)
        embedding = self.lrelu(self.fc1(self.dropout(x)))
        logits = self.fc2(embedding)
        return embedding, logits

    def _extract_mfcc(self, waveforms: Tensor) -> Tensor:
        x = waveforms.float()
        mfcc = self.mfcc(x)
        return normalize_mfcc_features(mfcc)

    def _prepare_cached_mfcc(self, mfcc_features: Tensor) -> Tensor:
        if mfcc_features.ndim == 2:
            mfcc_features = mfcc_features.unsqueeze(0)
        x = mfcc_features.float().transpose(1, 2)
        return normalize_mfcc_features(x)


class PyAraAASISTHead(nn.Module):
    def __init__(self, input_dim: int, *, projected_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        filts = [[1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0]

        self.input_proj = nn.Linear(input_dim, projected_dim)
        spectral_nodes = (projected_dim - 3) // 3 + 1

        self.first_bn = SafeBatchNorm2d(num_features=1)
        self.first_bn1 = SafeBatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            ResidualBlock(filts[0], first=True),
            ResidualBlock(filts[1]),
            ResidualBlock(filts[2]),
            ResidualBlock(filts[3]),
            ResidualBlock(filts[3]),
            ResidualBlock(filts[3]),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            SafeBatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        self.pos_s = nn.Parameter(torch.randn(1, spectral_nodes, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.gat_layer_s = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.gat_layer_t = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])
        self.htrg_layer_st11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.htrg_layer_st12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.htrg_layer_st21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.htrg_layer_st22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_s = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_t = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hs1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_ht1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hs2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_ht2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], num_classes)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        x = self.input_proj(features)
        x = x.transpose(1, 2).unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))
        x = self.encoder(x)
        x = self.selu(self.first_bn1(x))

        attention = self.attention(x)
        spectral_weights = F.softmax(attention, dim=-1)
        spectral = torch.sum(x * spectral_weights, dim=-1).transpose(1, 2)
        spectral = spectral + self.pos_s[:, : spectral.size(1), :]
        out_s = self.pool_s(self.gat_layer_s(spectral))

        temporal_weights = F.softmax(attention, dim=-2)
        temporal = torch.sum(x * temporal_weights, dim=-2).transpose(1, 2)
        out_t = self.pool_t(self.gat_layer_t(temporal))

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_t1, out_s1, master1 = self.htrg_layer_st11(out_t, out_s, master=master1)
        out_s1 = self.pool_hs1(out_s1)
        out_t1 = self.pool_ht1(out_t1)
        out_t_aug, out_s_aug, master_aug = self.htrg_layer_st12(out_t1, out_s1, master=master1)
        out_t1 = out_t1 + out_t_aug
        out_s1 = out_s1 + out_s_aug
        master1 = master1 + master_aug

        out_t2, out_s2, master2 = self.htrg_layer_st21(out_t, out_s, master=master2)
        out_s2 = self.pool_hs2(out_s2)
        out_t2 = self.pool_ht2(out_t2)
        out_t_aug, out_s_aug, master_aug = self.htrg_layer_st22(out_t2, out_s2, master=master2)
        out_t2 = out_t2 + out_t_aug
        out_s2 = out_s2 + out_s_aug
        master2 = master2 + master_aug

        out_t = torch.max(self.drop_way(out_t1), self.drop_way(out_t2))
        out_s = torch.max(self.drop_way(out_s1), self.drop_way(out_s2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))

        t_max, _ = torch.max(torch.abs(out_t), dim=1)
        t_avg = torch.mean(out_t, dim=1)
        s_max, _ = torch.max(torch.abs(out_s), dim=1)
        s_avg = torch.mean(out_s, dim=1)
        embedding = torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)
        logits = self.out_layer(self.drop(embedding))
        return embedding, logits


class Wav2VecPyAraSpoofDetector(nn.Module):
    def __init__(
        self,
        bundle_name: str = "WAV2VEC2_XLSR_300M",
        *,
        freeze_wav2vec: bool = False,
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 0,
        wav2vec_layers: int | None = None,
        projected_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.frontend = Wav2VecFrontend(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
        )
        self.head = PyAraAASISTHead(
            input_dim=self.frontend.out_dim,
            projected_dim=projected_dim,
            num_classes=num_classes,
        )

    @property
    def sample_rate(self) -> int:
        return self.frontend.sample_rate

    def forward(
        self,
        waveforms: Tensor | None,
        lengths: Tensor | None = None,
        *,
        ssl_features: Tensor | None = None,
        ssl_feature_lengths: Tensor | None = None,
        mfcc_features: Tensor | None = None,
        mfcc_feature_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del ssl_feature_lengths, mfcc_features, mfcc_feature_lengths
        if ssl_features is None:
            if waveforms is None:
                raise ValueError("waveforms are required when ssl_features are not provided")
            features, _ = self.frontend(waveforms, lengths)
        else:
            features = ssl_features
        return self.head(features)


class Wav2VecTemporalSpoofDetector(nn.Module):
    def __init__(
        self,
        bundle_name: str = "WAV2VEC2_XLSR_300M",
        *,
        freeze_wav2vec: bool = False,
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 0,
        wav2vec_layers: int | None = None,
        temporal_dim: int = 256,
        embedding_dim: int = 160,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.frontend = Wav2VecFrontend(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
        )
        self.head = Wav2VecTemporalHead(
            input_dim=self.frontend.out_dim,
            model_dim=temporal_dim,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

    @property
    def sample_rate(self) -> int:
        return self.frontend.sample_rate

    def forward(
        self,
        waveforms: Tensor | None,
        lengths: Tensor | None = None,
        *,
        ssl_features: Tensor | None = None,
        ssl_feature_lengths: Tensor | None = None,
        mfcc_features: Tensor | None = None,
        mfcc_feature_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del mfcc_features, mfcc_feature_lengths
        if ssl_features is None:
            if waveforms is None:
                raise ValueError("waveforms are required when ssl_features are not provided")
            features, feature_lengths = self.frontend(waveforms, lengths)
        else:
            features = ssl_features
            feature_lengths = ssl_feature_lengths
        return self.head(features, feature_lengths)


class MFCCResNetSpoofDetector(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.branch = MFCCResNetBranch(sample_rate=sample_rate, num_classes=num_classes)

    @property
    def sample_rate(self) -> int:
        return self.branch.sample_rate

    def forward(
        self,
        waveforms: Tensor | None,
        lengths: Tensor | None = None,
        *,
        ssl_features: Tensor | None = None,
        ssl_feature_lengths: Tensor | None = None,
        mfcc_features: Tensor | None = None,
        mfcc_feature_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del ssl_features, ssl_feature_lengths
        return self.branch(
            waveforms,
            lengths,
            mfcc_features=mfcc_features,
            mfcc_feature_lengths=mfcc_feature_lengths,
        )


class FusionSpoofDetector(nn.Module):
    def __init__(
        self,
        bundle_name: str = "WAV2VEC2_XLSR_300M",
        *,
        freeze_wav2vec: bool = False,
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 0,
        wav2vec_layers: int | None = None,
        projected_dim: int = 128,
        fusion_hidden_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.wav2vec_branch = Wav2VecPyAraSpoofDetector(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
            projected_dim=projected_dim,
            num_classes=num_classes,
        )
        self.mfcc_branch = MFCCResNetBranch(sample_rate=self.wav2vec_branch.sample_rate, num_classes=num_classes)
        self.embedding_dim = 160 + 128
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embedding_dim, fusion_hidden_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    @property
    def sample_rate(self) -> int:
        return self.wav2vec_branch.sample_rate

    def forward(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = None,
        *,
        ssl_features: Tensor | None = None,
        ssl_feature_lengths: Tensor | None = None,
        mfcc_features: Tensor | None = None,
        mfcc_feature_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        wav_embedding, _ = self.wav2vec_branch(
            waveforms,
            lengths,
            ssl_features=ssl_features,
            ssl_feature_lengths=ssl_feature_lengths,
        )
        mfcc_embedding, _ = self.mfcc_branch(
            waveforms,
            lengths,
            mfcc_features=mfcc_features,
            mfcc_feature_lengths=mfcc_feature_lengths,
        )
        embedding = torch.cat([wav_embedding, mfcc_embedding], dim=1)
        logits = self.fusion_classifier(embedding)
        return embedding, logits


class FusionTemporalSpoofDetector(nn.Module):
    def __init__(
        self,
        bundle_name: str = "WAV2VEC2_XLSR_300M",
        *,
        freeze_wav2vec: bool = False,
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 0,
        wav2vec_layers: int | None = None,
        temporal_dim: int = 256,
        temporal_embedding_dim: int = 160,
        fusion_hidden_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.wav2vec_branch = Wav2VecTemporalSpoofDetector(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
            temporal_dim=temporal_dim,
            embedding_dim=temporal_embedding_dim,
            num_classes=num_classes,
        )
        self.mfcc_branch = MFCCResNetBranch(sample_rate=self.wav2vec_branch.sample_rate, num_classes=num_classes)
        self.embedding_dim = temporal_embedding_dim + 128
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embedding_dim, fusion_hidden_dim),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    @property
    def sample_rate(self) -> int:
        return self.wav2vec_branch.sample_rate

    def forward(
        self,
        waveforms: Tensor,
        lengths: Tensor | None = None,
        *,
        ssl_features: Tensor | None = None,
        ssl_feature_lengths: Tensor | None = None,
        mfcc_features: Tensor | None = None,
        mfcc_feature_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        wav_embedding, _ = self.wav2vec_branch(
            waveforms,
            lengths,
            ssl_features=ssl_features,
            ssl_feature_lengths=ssl_feature_lengths,
        )
        mfcc_embedding, _ = self.mfcc_branch(
            waveforms,
            lengths,
            mfcc_features=mfcc_features,
            mfcc_feature_lengths=mfcc_feature_lengths,
        )
        embedding = torch.cat([wav_embedding, mfcc_embedding], dim=1)
        logits = self.fusion_classifier(embedding)
        return embedding, logits


def build_spoof_detector(
    model_type: str = "fusion",
    *,
    bundle_name: str = "WAV2VEC2_XLSR_300M",
    freeze_wav2vec: bool = False,
    freeze_feature_extractor: bool = True,
    freeze_transformer_layers: int = 0,
    wav2vec_layers: int | None = None,
    num_classes: int = 2,
) -> nn.Module:
    if model_type == "wav2vec_pyara":
        return Wav2VecPyAraSpoofDetector(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
            num_classes=num_classes,
        )
    if model_type == "wav2vec_temporal":
        return Wav2VecTemporalSpoofDetector(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
            num_classes=num_classes,
        )
    if model_type == "mfcc_resnet":
        return MFCCResNetSpoofDetector(num_classes=num_classes)
    if model_type == "fusion":
        return FusionSpoofDetector(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
            num_classes=num_classes,
        )
    if model_type == "fusion_temporal":
        return FusionTemporalSpoofDetector(
            bundle_name=bundle_name,
            freeze_wav2vec=freeze_wav2vec,
            freeze_feature_extractor=freeze_feature_extractor,
            freeze_transformer_layers=freeze_transformer_layers,
            wav2vec_layers=wav2vec_layers,
            num_classes=num_classes,
        )
    raise ValueError(
        "model_type must be one of: fusion, fusion_temporal, wav2vec_pyara, wav2vec_temporal, mfcc_resnet"
    )
