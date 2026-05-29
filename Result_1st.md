# Phase 2 Deep Learning - Experimental Results

## Summary

This document summarizes the results from Phase 2 deep learning experiments using CIFAR-10 dataset.

## Experiments Conducted

### 1. PyTorch Fundamentals (buoi1_tensor_autograd.py)
- Validated tensor operations and autograd functionality
- Tested Conv2d layer with proper shape calculations
- Device: CPU
- Status: ✅ Completed successfully

### 2. Dataset Visualization
- Downloaded and processed CIFAR-10 dataset (170M)
- Dataset size: 50,000 training samples + 10,000 test samples
- Image dimensions: 28×28 pixels (after normalization)
- Status: ✅ Dataset prepared successfully

### 3. Experiment A: CNN Model (4-layer Convolutional Network)

**Configuration:** `configs/phase2_cnn.yaml`

**Training Results (10 epochs):**
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|----------|----------|---------|
| 1 | 1.6286 | 39.62% | 1.4358 | 47.90% |
| 2 | 1.3776 | 49.77% | 1.2429 | 54.96% |
| 3 | 1.2674 | 54.00% | 1.1902 | 58.22% |
| 4 | 1.2070 | 56.90% | 1.0871 | 60.12% |
| 5 | 1.1602 | 58.63% | 1.0402 | 63.44% |
| 6 | 1.1235 | 59.99% | 1.0020 | 64.88% |
| 7 | 1.0895 | 61.40% | 0.9776 | 65.46% |
| 8 | 1.0682 | 62.34% | 0.9806 | 66.02% |
| 9 | 1.0476 | 63.10% | 0.9039 | **68.38%** ⭐ |
| 10 | 1.0192 | 64.21% | 0.9366 | 67.30% |

**Best Validation Accuracy:** 68.38% (Epoch 9)

**Test Set Evaluation:**
- **Test Accuracy:** 71.91%
- **Macro F1-Score:** 0.7133
- Checkpoint: `outputs/phase2/cnn_best.pt`
- Training Time: ~27-29s per epoch
- **Status:** ✅ Best performing model

### 4. Experiment B: ResNet18 (Pre-trained ImageNet weights)

**Configuration:** `configs/phase2_resnet18.yaml`

**Training Results (10 epochs):**
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|----------|----------|---------|
| 1 | 1.7278 | 38.47% | 1.5763 | 44.54% |
| 2 | 1.5788 | 44.47% | 1.5657 | 44.38% |
| 3 | 1.5525 | 45.31% | 1.5356 | 45.54% |
| 4 | 1.5466 | 45.44% | 1.5213 | 45.56% |
| 5 | 1.5369 | 45.84% | 1.5525 | 45.86% |
| 6 | 1.5419 | 45.77% | 1.5194 | 46.48% |
| 7 | 1.5344 | 46.13% | 1.5243 | 45.98% |
| 8 | 1.5290 | 46.34% | 1.5052 | 46.66% |
| 9 | 1.5405 | 45.99% | 1.5117 | 46.34% |
| 10 | 1.5320 | 46.22% | 1.5021 | **46.70%** |

**Best Validation Accuracy:** 46.70% (Epoch 10)

**Observations:**
- Training Time: ~247-275s per epoch (much slower than CNN)
- Convergence plateaued early (~epoch 5+)
- Significantly underperformed compared to CNN model
- Pre-trained ImageNet weights did not transfer well to CIFAR-10
- **Status:** ⚠️ Suboptimal performance

## Comparison

| Metric | CNN | ResNet18 |
|--------|-----|----------|
| Best Val Accuracy | 68.38% | 46.70% |
| Test Accuracy | 71.91% | - |
| Final Train Loss | 1.0192 | 1.5320 |
| Final Val Loss | 0.9366 | 1.5021 |
| Time per Epoch | ~28s | ~250s |
| Macro F1 (Test) | 0.7133 | - |

## Key Findings

1. **CNN Model Performance**: The custom 4-layer CNN achieved superior performance with:
   - 68.38% best validation accuracy
   - 71.91% test accuracy
   - Rapid training convergence
   - Efficient computation (28s/epoch)

2. **ResNet18 Limitations**: Pre-trained ResNet18 did not generalize well:
   - Plateaued at ~46% accuracy
   - ImageNet pre-training may have hindered CIFAR-10 adaptation
   - Computational overhead not justified by performance gains

3. **Recommendations**:
   - Use CNN model for production/deployment
   - Consider fine-tuning strategies for ResNet18 (lower learning rate, gradual unfreezing)
   - Explore data augmentation techniques to improve generalization
   - Investigate ensemble methods combining both models

## Environment

- **Device:** CPU
- **Framework:** PyTorch
- **Dataset:** CIFAR-10
- **Number of Classes:** 10
- **Python Version:** 3.12

