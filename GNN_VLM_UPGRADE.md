# Innovation Point 1: VLM-Guided Heterogeneous GNN

This upgrade makes the graph branch stronger than the earlier pixel-statistics-only head.

## What changed

1. Spatial nodes now fuse BLIP2 vision encoder patch tokens with RGB patch statistics.
2. Frequency nodes use FFT statistics plus high-frequency residual statistics.
3. Learnable forensic concept nodes are added as text-side graph nodes.
4. Graph topology contains local 8-neighbor edges and non-local semantic kNN edges.
5. The graph head uses attention readout instead of plain average pooling.
6. Training can isolate innovation point 1 with `--gnn_short_answer`, so CoT does not confound the accuracy ablation.

## Fast sanity check on AutoDL

Run this in the `(vlm_detect)` environment:

```bash
cd /root/autodl-tmp/project/VLM-DETECT-main

python - <<'PY'
import torch
from gnn_cot import GNNCoTConfig, HeteroGNNCoTHead

config = GNNCoTConfig(
    vocab_size=1000,
    hidden_dim=64,
    text_max_nodes=8,
    image_grid_size=4,
    num_layers=1,
    num_heads=4,
    visual_feature_dim=1408,
    forensic_nodes=4,
    semantic_top_k=2,
)
head = HeteroGNNCoTHead(config).cuda()
out = head(
    pixel_values=torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.float16),
    input_ids=torch.randint(0, 1000, (2, 16), device="cuda"),
    attention_mask=torch.ones(2, 16, device="cuda", dtype=torch.long),
    vision_embeds=torch.randn(2, 257, 1408, device="cuda", dtype=torch.float16),
    cls_labels=torch.tensor([0, 1], device="cuda"),
)
print(out["logits"].shape, out["loss"].item(), out["spatial_importance"].shape)
PY
```

Expected output shape:

```text
torch.Size([2, 2]) ... torch.Size([2, 4, 4])
```

## Step 1: train the original Bi-LORA baseline

```bash
python blip2_detect_aligned.py \
  --dataset ./data/Train_CSV_Balanced/train_LDM_balanced.csv \
  --base_model ./blip2-opt-2.7b \
  --epochs 20 \
  --batch_size 24 \
  --num_workers 4 \
  --save_path ./SaveFineTune/LDM-bilora-baseline \
  --cls_loss_weight 0.5
```

## Step 2: train innovation point 1 only

This is the cleanest accuracy ablation. It enables the graph branch but keeps the language target as short `fake/real`.

```bash
python blip2_detect_aligned.py \
  --dataset ./data/Train_CSV_Balanced/train_LDM_balanced.csv \
  --base_model ./blip2-opt-2.7b \
  --epochs 20 \
  --batch_size 16 \
  --num_workers 4 \
  --save_path ./SaveFineTune/LDM-vlm-gnn \
  --use_gnn_cot \
  --gnn_short_answer \
  --gnn_grid_size 4 \
  --gnn_hidden_dim 256 \
  --gnn_layers 2 \
  --gnn_heads 4 \
  --gnn_forensic_nodes 8 \
  --gnn_semantic_top_k 4 \
  --gnn_loss_weight 0.4 \
  --cls_loss_weight 0.5 \
  --rlhf_reward_weight 0
```

For a 24 GB RTX 4090, start with `--batch_size 16`. If memory is stable, try `24`; if OOM, use `8` and keep all other settings unchanged.

## Step 3: evaluate baseline and GNN fairly

Baseline:

```bash
python blip2_test004.py \
  --base_model ./blip2-opt-2.7b \
  --model_path ./SaveFineTune/LDM-bilora-baseline/epoch020 \
  --dataset ./data/Test_CSV/test_LDM.csv \
            ./data/Test_CSV/test_ADM.csv \
            ./data/Test_CSV/test_DDPM.csv \
            ./data/Test_CSV/test_IDDPM.csv \
            ./data/Test_CSV/test_PNDM.csv \
            ./data/Test_CSV/test_ProGAN.csv \
            ./data/Test_CSV/test_StyleGAN.csv \
            ./data/Test_CSV/test_Diff-StyleGAN2.csv \
            ./data/Test_CSV/test_Diff-ProjectedGAN.csv \
  --decision_source lm \
  --score_mode short
```

GNN fusion:

```bash
python blip2_test004.py \
  --base_model ./blip2-opt-2.7b \
  --model_path ./SaveFineTune/LDM-vlm-gnn/epoch020 \
  --gnn_head_path ./SaveFineTune/LDM-vlm-gnn/epoch020/gnn_cot_head.pt \
  --dataset ./data/Test_CSV/test_LDM.csv \
            ./data/Test_CSV/test_ADM.csv \
            ./data/Test_CSV/test_DDPM.csv \
            ./data/Test_CSV/test_IDDPM.csv \
            ./data/Test_CSV/test_PNDM.csv \
            ./data/Test_CSV/test_ProGAN.csv \
            ./data/Test_CSV/test_StyleGAN.csv \
            ./data/Test_CSV/test_Diff-StyleGAN2.csv \
            ./data/Test_CSV/test_Diff-ProjectedGAN.csv \
  --decision_source fusion \
  --gnn_vote_weight 0.35 \
  --score_mode short
```

## Step 4: ablations for the paper

Use these runs to prove the gain comes from the graph design:

```bash
# no BLIP2 vision-token node feature, only pixel/frequency graph
python blip2_detect_aligned.py \
  --dataset ./data/Train_CSV_Balanced/train_LDM_balanced.csv \
  --base_model ./blip2-opt-2.7b \
  --epochs 20 \
  --batch_size 16 \
  --save_path ./SaveFineTune/LDM-gnn-no-vlm-token \
  --use_gnn_cot \
  --gnn_short_answer \
  --gnn_no_vision_tokens \
  --rlhf_reward_weight 0

# no non-local semantic edges
python blip2_detect_aligned.py \
  --dataset ./data/Train_CSV_Balanced/train_LDM_balanced.csv \
  --base_model ./blip2-opt-2.7b \
  --epochs 20 \
  --batch_size 16 \
  --save_path ./SaveFineTune/LDM-gnn-no-knn-edge \
  --use_gnn_cot \
  --gnn_short_answer \
  --gnn_semantic_top_k 0 \
  --rlhf_reward_weight 0
```

Recommended table:

```text
Bi-LORA baseline
Bi-LORA + pixel/frequency GNN
Bi-LORA + VLM-token GNN
Bi-LORA + VLM-token GNN + semantic kNN edges
Bi-LORA + VLM-token GNN + semantic kNN edges + CoT/RLHF
```
