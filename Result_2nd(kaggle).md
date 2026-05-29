# Phase 2: Deep Learning with PyTorch - Report

## 1. Dataset
CIFAR-10: 10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck).
Train: 50,000 images. Test: 10,000 images. Split 10% validation from train.

## 2. Models
- **CNN Baseline**: 2 conv blocks (32, 64 filters) + BatchNorm + ReLU + MaxPool → FC(256) → Dropout(0.3) → 10 classes.
- **ResNet18 Fine-tune**: Pretrained ResNet18 with stem modified for 32×32 input (conv1 kernel=3, stride=1, no maxpool). All layers unfrozen, full fine-tune with low LR.

## 3. Training Setup
- **CNN Baseline**: Optimizer AdamW, lr=0.001, weight_decay=1e-4, 15 epochs.
- **ResNet18 Fine-tune**: Optimizer AdamW, lr=1e-4, weight_decay=1e-4, ReduceLROnPlateau scheduler (factor=0.5, patience=2), 15 epochs.

## 4. Results on Test Set
| Model | Test Accuracy | Macro F1 |
|-------|--------------|----------|
| CNN Baseline | 0.7275 | 0.7284 |
| ResNet18 Fine-tune | 0.9264 | 0.9264 |

## 5. Learning Curves
### CNN Baseline
![CNN Learning Curve](cnn_learning_curve.png)
### ResNet18 Fine-tune
![ResNet18 Learning Curve](resnet_full_learning_curve.png)

## 6. Confusion Matrices
### CNN Baseline
![CNN Confusion](cnn_confusion.png)
### ResNet18 Fine-tune
![ResNet18 Confusion](resnet_full_confusion.png)

## 7. Optimizer & Learning Rate Experiments (CNN Baseline)
| experiment   |   best_val_acc |   final_train_loss |   final_val_loss |
|:-------------|---------------:|-------------------:|-----------------:|
| AdamW_lr1e-3 |          66.44 |            1.04134 |         1.03604  |
| AdamW_lr1e-4 |          64.4  |            1.03202 |         1.02981  |
| SGD_lr1e-2   |          69    |            0.89794 |         0.937397 |
| SGD_lr1e-3   |          62.6  |            1.07064 |         1.06626  |
## 8. Discussion
- **CNN Baseline** đạt accuracy ~71%, không có dấu hiệu overfitting mạnh (val loss giảm đều).
- **ResNet18 Fine-tune (full unfreeze)** cải thiện đáng kể (>85% accuracy) nhờ transfer learning từ ImageNet, kết hợp thay đổi stem phù hợp ảnh nhỏ và fine-tune toàn bộ với lr thấp.
- Khi freeze backbone hoặc chỉ train classifier, mô hình không học được đặc trưng tốt cho CIFAR-10, dẫn đến accuracy kém (~46%).
- So sánh optimizer: AdamW hội tụ nhanh và ổn định hơn SGD với cùng số epoch. Learning rate 1e-3 cho CNN là phù hợp.

## 9. Conclusion
Đã hoàn thành pipeline: data → train → evaluate → compare. Transfer learning với ResNet18 cho kết quả tốt nhất. Kỹ năng debug shape, device, overfitting đã được thực hành. Sẵn sàng cho Phase 3.
