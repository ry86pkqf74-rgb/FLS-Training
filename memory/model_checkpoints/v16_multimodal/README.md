# v16 Multimodal LoRA Training Results

## Summary
- **Model**: Qwen2.5-VL-7B-Instruct + LoRA (r=32, alpha=32)
- **Type**: Multimodal (8 video frames + text per example)
- **Train loss**: 0.3332
- **Eval loss**: 0.2234
- **Steps**: 216 (3 epochs)
- **Hardware**: NVIDIA H100 80GB SXM (RunPod), ~80 min

## Dataset
- 1,151 train / 141 val examples
- Task distribution: task2=534, task1=295, task5=281, task3=16, task4=16, task6=9
- Sources: LASANA (1013), YouTube consensus (114), user practice (24)
- 8 sampled frames per video at reduced resolution

## Key Technical Details
- Custom Trainer (not SFTTrainer) with MultimodalDataCollator
- Label masking: only compute loss on assistant response tokens
- Gradient checkpointing fix: `model.enable_input_require_grads()` + `use_reentrant=False`
- 4-bit NF4 quantization with bitsandbytes during training
- Image resolution: 128-256 × 28×28 pixels per frame
- MAX_LENGTH=4096 tokens (8 images × ~256 tokens + text)

## Comparison with v15 (text-only)
| Metric | v15 (text) | v16 (multimodal) |
|--------|-----------|-----------------|
| Train loss | 0.2921 | 0.3332 |
| Eval loss | 0.1775 | 0.2234 |
| Examples | 1,433 | 1,151 |
| Epochs | 4 | 3 |
| Time | ~8 min | ~80 min |

## Adapter Locations
- S8 backup: `/data/fls/adapters/v16/`
- S6 demo (ACTIVE): `/opt/fls/adapters/v16/`
- Demo URL: http://38.242.238.209:7860
