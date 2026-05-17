import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNCoTConfig:
    vocab_size: int
    hidden_dim: int = 256
    text_max_nodes: int = 24
    image_grid_size: int = 4
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    visual_feature_dim: int = 1408
    forensic_nodes: int = 8
    semantic_top_k: int = 4


class PatchGraphEncoder(nn.Module):
    """Build spatial, frequency, and text nodes for a small heterogeneous graph."""

    def __init__(self, config: GNNCoTConfig):
        super().__init__()
        self.config = config
        self.spatial_proj = nn.Linear(6, config.hidden_dim)
        self.vision_proj = nn.Linear(config.visual_feature_dim, config.hidden_dim)
        self.spatial_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Sigmoid(),
        )
        self.frequency_proj = nn.Linear(12, config.hidden_dim)
        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.forensic_nodes = nn.Parameter(torch.randn(config.forensic_nodes, config.hidden_dim) * 0.02)
        self.type_embedding = nn.Embedding(3, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.image_grid_size * config.image_grid_size, config.hidden_dim)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
    ):
        pixel_values = pixel_values.float()
        batch_size = pixel_values.shape[0]
        grid = self.config.image_grid_size

        spatial_stats = self._patch_stats(pixel_values, grid)
        spatial = self.spatial_proj(spatial_stats)
        if vision_embeds is not None:
            vision = self._vision_grid(vision_embeds.float(), grid)
            vision = self.vision_proj(vision)
            gate = self.spatial_gate(torch.cat([spatial, vision], dim=-1))
            spatial = gate * vision + (1.0 - gate) * spatial

        fft_map = torch.fft.fft2(pixel_values, norm="ortho")
        freq = torch.log1p(torch.abs(torch.fft.fftshift(fft_map, dim=(-2, -1))))
        low = F.interpolate(
            F.avg_pool2d(pixel_values, kernel_size=5, stride=1, padding=2),
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_residual = (pixel_values - low).abs()
        freq = torch.cat([self._patch_stats(freq, grid), self._patch_stats(high_residual, grid)], dim=-1)
        freq = self.frequency_proj(freq)

        position_ids = torch.arange(grid * grid, device=pixel_values.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        spatial = spatial + self.position_embedding(position_ids)
        freq = freq + self.position_embedding(position_ids)

        text_ids = input_ids[:, : self.config.text_max_nodes].clamp(
            min=0,
            max=self.text_embedding.num_embeddings - 1,
        )
        text_mask = attention_mask[:, : text_ids.shape[1]].bool()
        text = self.text_embedding(text_ids)
        if self.config.forensic_nodes > 0:
            forensic = self.forensic_nodes.unsqueeze(0).expand(batch_size, -1, -1)
            forensic_mask = torch.ones(
                batch_size,
                self.config.forensic_nodes,
                dtype=torch.bool,
                device=text.device,
            )
            text = torch.cat([forensic, text], dim=1)
            text_mask = torch.cat([forensic_mask, text_mask], dim=1)

        spatial_type = torch.zeros(spatial.shape[:2], dtype=torch.long, device=spatial.device)
        freq_type = torch.ones(freq.shape[:2], dtype=torch.long, device=freq.device)
        text_type = torch.full(text.shape[:2], 2, dtype=torch.long, device=text.device)

        spatial = spatial + self.type_embedding(spatial_type)
        freq = freq + self.type_embedding(freq_type)
        text = text + self.type_embedding(text_type)

        nodes = torch.cat([spatial, freq, text], dim=1)
        node_mask = torch.cat(
            [
                torch.ones(spatial.shape[:2], dtype=torch.bool, device=nodes.device),
                torch.ones(freq.shape[:2], dtype=torch.bool, device=nodes.device),
                text_mask,
            ],
            dim=1,
        )

        edge_types, edge_mask = self._build_graph_topology(batch_size, grid, text.shape[1], nodes.device)
        if self.config.semantic_top_k > 0:
            image_nodes = grid * grid
            edge_types = edge_types.clone()
            edge_mask = edge_mask.clone()
            self._add_semantic_knn_edges(
                edge_types=edge_types,
                edge_mask=edge_mask,
                features=spatial,
                start=0,
                top_k=self.config.semantic_top_k,
            )
            self._add_semantic_knn_edges(
                edge_types=edge_types,
                edge_mask=edge_mask,
                features=freq,
                start=image_nodes,
                top_k=self.config.semantic_top_k,
            )
        return nodes, node_mask, edge_types, edge_mask

    @staticmethod
    def _patch_stats(values: torch.Tensor, grid: int):
        mean = F.adaptive_avg_pool2d(values, (grid, grid))
        sq_mean = F.adaptive_avg_pool2d(values.square(), (grid, grid))
        std = (sq_mean - mean.square()).clamp_min(1e-6).sqrt()
        stats = torch.cat([mean, std], dim=1)
        return stats.flatten(2).transpose(1, 2)

    @staticmethod
    def _vision_grid(vision_embeds: torch.Tensor, grid: int):
        batch_size, num_tokens, hidden_dim = vision_embeds.shape
        patch_tokens = vision_embeds
        patch_count = num_tokens - 1
        patch_side = int(math.sqrt(patch_count))
        if patch_count > 0 and patch_side * patch_side == patch_count:
            patch_tokens = vision_embeds[:, 1:, :]
            num_tokens = patch_count
        else:
            patch_side = int(math.sqrt(num_tokens))

        if patch_side * patch_side == num_tokens:
            patch_map = patch_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, patch_side, patch_side)
            pooled = F.adaptive_avg_pool2d(patch_map, (grid, grid))
            return pooled.flatten(2).transpose(1, 2)

        pooled = F.adaptive_avg_pool1d(patch_tokens.transpose(1, 2), grid * grid)
        return pooled.transpose(1, 2)

    @staticmethod
    def _local_grid_mask(grid: int, device: torch.device):
        num_nodes = grid * grid
        mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
        for row in range(grid):
            for col in range(grid):
                src = row * grid + col
                for d_row in (-1, 0, 1):
                    for d_col in (-1, 0, 1):
                        dst_row = row + d_row
                        dst_col = col + d_col
                        if 0 <= dst_row < grid and 0 <= dst_col < grid:
                            dst = dst_row * grid + dst_col
                            mask[src, dst] = True
        return mask

    @staticmethod
    def _add_semantic_knn_edges(
        edge_types: torch.Tensor,
        edge_mask: torch.Tensor,
        features: torch.Tensor,
        start: int,
        top_k: int,
    ):
        batch_size, num_nodes, _ = features.shape
        k = max(1, min(top_k + 1, num_nodes))
        normed = F.normalize(features.float(), dim=-1)
        similarity = torch.matmul(normed, normed.transpose(1, 2))
        knn = similarity.topk(k=k, dim=-1).indices
        batch_index = torch.arange(batch_size, device=features.device).view(batch_size, 1, 1)
        src_index = torch.arange(num_nodes, device=features.device).view(1, num_nodes, 1)
        src_index = src_index.expand(batch_size, num_nodes, k) + start
        dst_index = knn + start
        edge_mask[batch_index, src_index, dst_index] = True
        edge_types[batch_index, src_index, dst_index] = 6

    @classmethod
    def _build_graph_topology(cls, batch_size: int, grid: int, text_nodes: int, device: torch.device):
        image_nodes = grid * grid
        total_nodes = image_nodes * 2 + text_nodes
        edge_types = torch.zeros(total_nodes, total_nodes, dtype=torch.long, device=device)
        edge_mask = torch.zeros(total_nodes, total_nodes, dtype=torch.bool, device=device)

        spatial_slice = slice(0, image_nodes)
        freq_slice = slice(image_nodes, image_nodes * 2)
        text_slice = slice(image_nodes * 2, total_nodes)

        local_grid = cls._local_grid_mask(grid, device)

        edge_types[spatial_slice, spatial_slice] = 0
        edge_mask[spatial_slice, spatial_slice] = local_grid

        edge_types[freq_slice, freq_slice] = 1
        edge_mask[freq_slice, freq_slice] = local_grid

        edge_types[text_slice, text_slice] = 2
        edge_mask[text_slice, text_slice] = True

        edge_types[spatial_slice, freq_slice] = 3
        edge_types[freq_slice, spatial_slice] = 3
        edge_mask[spatial_slice, freq_slice] = local_grid
        edge_mask[freq_slice, spatial_slice] = local_grid

        edge_types[spatial_slice, text_slice] = 4
        edge_types[text_slice, spatial_slice] = 4
        edge_mask[spatial_slice, text_slice] = True
        edge_mask[text_slice, spatial_slice] = True

        edge_types[freq_slice, text_slice] = 5
        edge_types[text_slice, freq_slice] = 5
        edge_mask[freq_slice, text_slice] = True
        edge_mask[text_slice, freq_slice] = True

        edge_types = edge_types.unsqueeze(0).expand(batch_size, -1, -1)
        edge_mask = edge_mask.unsqueeze(0).expand(batch_size, -1, -1)
        return edge_types, edge_mask


class HeterogeneousGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_edge_types: int = 7):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Embedding(num_edge_types, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        node_mask: torch.Tensor,
        edge_types: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, num_nodes, hidden_dim = nodes.shape

        q = self.q_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(nodes).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_types).permute(0, 3, 1, 2)

        key_mask = node_mask[:, None, None, :]
        if edge_mask is None:
            allowed_mask = key_mask
        else:
            allowed_mask = edge_mask[:, None, :, :] & key_mask
            has_key = allowed_mask.any(dim=-1, keepdim=True)
            allowed_mask = torch.where(has_key, allowed_mask, key_mask.expand_as(allowed_mask))

        scores = scores.masked_fill(~allowed_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        messages = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, num_nodes, hidden_dim)
        nodes = self.norm1(nodes + self.dropout(self.out_proj(messages)))
        nodes = self.norm2(nodes + self.dropout(self.ffn(nodes)))
        nodes = nodes * node_mask.unsqueeze(-1)
        return nodes, attn


class HeteroGNNCoTHead(nn.Module):
    """Auxiliary graph fusion head for fake/real detection and CoT alignment."""

    def __init__(self, config: GNNCoTConfig):
        super().__init__()
        self.config = config
        self.encoder = PatchGraphEncoder(config)
        self.layers = nn.ModuleList(
            [
                HeterogeneousGraphLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_labels: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        nodes, node_mask, edge_types, edge_mask = self.encoder(
            pixel_values,
            input_ids,
            attention_mask,
            vision_embeds=vision_embeds,
        )
        last_attn = None
        for layer in self.layers:
            nodes, last_attn = layer(nodes, node_mask, edge_types, edge_mask)

        readout_scores = self.readout(nodes).squeeze(-1).masked_fill(~node_mask, torch.finfo(nodes.dtype).min)
        readout_weights = torch.softmax(readout_scores, dim=-1)
        pooled = torch.sum(nodes * readout_weights.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
        output = {"logits": logits, "graph_repr": pooled, "readout_weights": readout_weights}
        if last_attn is not None:
            output["attn"] = last_attn
            attention_importance = self._node_importance(last_attn, node_mask)
            output["node_importance"] = 0.5 * attention_importance + 0.5 * readout_weights
            image_nodes = self.config.image_grid_size * self.config.image_grid_size
            grid = self.config.image_grid_size
            output["spatial_importance"] = output["node_importance"][:, :image_nodes].reshape(-1, grid, grid)
            output["frequency_importance"] = output["node_importance"][:, image_nodes : image_nodes * 2].reshape(
                -1,
                grid,
                grid,
            )
        if cls_labels is not None:
            output["loss"] = F.cross_entropy(logits.float(), cls_labels.long())
            output["accuracy"] = (logits.argmax(dim=-1) == cls_labels).float().mean()
        return output

    @staticmethod
    def _node_importance(attn: torch.Tensor, node_mask: torch.Tensor):
        query_mask = node_mask[:, None, :, None].float()
        importance = (attn * query_mask).sum(dim=(1, 2))
        importance = importance / query_mask.sum(dim=(1, 2)).clamp_min(1.0)
        importance = importance * node_mask.float()
        return importance / importance.sum(dim=1, keepdim=True).clamp_min(1e-6)


class StructuredCoTReward(nn.Module):
    """Differentiable reward proxy for structured CoT and visual-label alignment."""

    def __init__(
        self,
        tokenizer,
        final_weight: float = 0.4,
        structure_weight: float = 0.3,
        alignment_weight: float = 0.3,
        feedback_weight: float = 0.0,
    ):
        super().__init__()
        self.final_weight = final_weight
        self.structure_weight = structure_weight
        self.alignment_weight = alignment_weight
        self.feedback_weight = feedback_weight
        self.structure_token_ids = self._encode_many(
            tokenizer,
            ["Quick", "intuition", "Salient", "evidence", "Deep", "reasoning", "Final", "conclusion"],
        )
        self.fake_token_ids = self._encode_many(tokenizer, ["fake"])
        self.real_token_ids = self._encode_many(tokenizer, ["real"])

    @staticmethod
    def _encode_many(tokenizer, words: Iterable[str]):
        token_ids = []
        for word in words:
            encoded = tokenizer(word, add_special_tokens=False).input_ids
            token_ids.extend(encoded)
            encoded_with_space = tokenizer(" " + word, add_special_tokens=False).input_ids
            token_ids.extend(encoded_with_space)
        return sorted(set(int(token_id) for token_id in token_ids))

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        graph_logits: torch.Tensor,
        cls_labels: torch.Tensor,
        feedback_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        label_mask = shift_labels.ne(-100)
        sample_nll = (token_loss * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp_min(1)
        language_reward = torch.exp(-sample_nll.clamp(max=20.0))

        structure_mask = self._membership_mask(shift_labels, self.structure_token_ids) & label_mask
        structure_reward = self._masked_exp_reward(token_loss, structure_mask)

        final_reward = self._final_answer_reward(token_loss, shift_labels, cls_labels, label_mask)
        align_reward = graph_logits.softmax(dim=-1).gather(1, cls_labels.view(-1, 1)).squeeze(1)

        reward = (
            self.final_weight * final_reward
            + self.structure_weight * structure_reward
            + self.alignment_weight * align_reward
        )
        reward = 0.5 * reward + 0.5 * language_reward

        loss = 1.0 - reward
        feedback_reward = None
        if feedback_scores is not None and self.feedback_weight > 0:
            feedback_reward = feedback_scores.to(reward.device).float().view(-1).clamp(0.0, 1.0)
            sample_weight = 1.0 + self.feedback_weight * (1.0 - feedback_reward)
            loss = loss * sample_weight

        return {
            "loss": loss.mean(),
            "reward": reward.detach().mean(),
            "final_reward": final_reward.detach().mean(),
            "structure_reward": structure_reward.detach().mean(),
            "alignment_reward": align_reward.detach().mean(),
            "feedback_reward": feedback_reward.detach().mean() if feedback_reward is not None else None,
        }

    @staticmethod
    def _membership_mask(labels: torch.Tensor, token_ids):
        if not token_ids:
            return torch.zeros_like(labels, dtype=torch.bool)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in token_ids:
            mask = mask | labels.eq(token_id)
        return mask

    @staticmethod
    def _masked_exp_reward(token_loss: torch.Tensor, mask: torch.Tensor):
        denom = mask.sum(dim=1)
        masked_nll = (token_loss * mask).sum(dim=1) / denom.clamp_min(1)
        reward = torch.exp(-masked_nll.clamp(max=20.0))
        return torch.where(denom.gt(0), reward, torch.zeros_like(reward))

    def _final_answer_reward(self, token_loss, labels, cls_labels, label_mask):
        rewards = []
        for idx in range(labels.shape[0]):
            token_ids = self.fake_token_ids if int(cls_labels[idx].item()) == 1 else self.real_token_ids
            mask = self._membership_mask(labels[idx : idx + 1], token_ids).squeeze(0) & label_mask[idx]
            if mask.any():
                sample_nll = token_loss[idx][mask].mean()
                rewards.append(torch.exp(-sample_nll.clamp(max=20.0)))
            else:
                rewards.append(token_loss.new_tensor(0.0))
        return torch.stack(rewards)


def build_gnn_cot_head(
    tokenizer,
    hidden_dim: int,
    text_max_nodes: int,
    image_grid_size: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    visual_feature_dim: int = 1408,
    forensic_nodes: int = 8,
    semantic_top_k: int = 4,
):
    vocab_size = len(tokenizer)
    config = GNNCoTConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        text_max_nodes=text_max_nodes,
        image_grid_size=image_grid_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        visual_feature_dim=visual_feature_dim,
        forensic_nodes=forensic_nodes,
        semantic_top_k=semantic_top_k,
    )
    return HeteroGNNCoTHead(config)


def save_gnn_cot_head(head: HeteroGNNCoTHead, save_path: str):
    torch.save({"config": head.config.__dict__, "state_dict": head.state_dict()}, save_path)


def load_gnn_cot_head(save_path: str, tokenizer, map_location=None):
    payload = torch.load(save_path, map_location=map_location)
    config_dict = dict(payload["config"])
    config_dict.setdefault("vocab_size", len(tokenizer))
    head = HeteroGNNCoTHead(GNNCoTConfig(**config_dict))
    head.load_state_dict(payload["state_dict"])
    return head


def _unwrap_blip2_model(model):
    if hasattr(model, "get_base_model"):
        try:
            model = model.get_base_model()
        except TypeError:
            pass
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model = model.base_model.model
    if hasattr(model, "model") and hasattr(model.model, "vision_model"):
        model = model.model
    return model


def infer_blip2_vision_dim(model, default: int = 1408) -> int:
    core = _unwrap_blip2_model(model)
    config = getattr(core, "config", None)
    vision_config = getattr(config, "vision_config", None)
    hidden_size = getattr(vision_config, "hidden_size", None)
    return int(hidden_size) if hidden_size is not None else int(default)


def extract_blip2_vision_tokens(model, pixel_values: torch.Tensor) -> Optional[torch.Tensor]:
    core = _unwrap_blip2_model(model)
    vision_model = getattr(core, "vision_model", None)
    if vision_model is None:
        return None
    was_training = vision_model.training
    vision_model.eval()
    with torch.no_grad():
        outputs = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
    if was_training:
        vision_model.train()
    return outputs.last_hidden_state.detach()


def format_top_patches(graph_outputs: Dict[str, torch.Tensor], top_k: int = 3) -> List[str]:
    """Return compact top-patch evidence strings from spatial/frequency importance maps."""

    spatial = graph_outputs.get("spatial_importance")
    frequency = graph_outputs.get("frequency_importance")
    if spatial is None or frequency is None:
        return []

    combined = (spatial.detach().float().cpu() + frequency.detach().float().cpu()) * 0.5
    batch_size, grid, _ = combined.shape
    top_k = max(1, min(top_k, grid * grid))
    evidence = []
    for batch_idx in range(batch_size):
        values, indices = torch.topk(combined[batch_idx].flatten(), k=top_k)
        parts = []
        for value, index in zip(values.tolist(), indices.tolist()):
            row = index // grid
            col = index % grid
            parts.append(f"patch({row},{col})={value:.4f}")
        evidence.append("; ".join(parts))
    return evidence
