# Weekly Report - Week 2

**Intern:** Nguyễn Đức Nam Khánh 
**Thời gian:** 27/05/2026 - 30/05/2026  
**Phase hiện tại:** Phase 2 - Deep Learning with PyTorch  
**Mentor/Reviewer:** Nguyễn Bảo Anh

---

## 1. Mục Tiêu Tuần Này

Các mục tiêu đã đặt ra cho tuần này:

* Xây dựng pipeline huấn luyện ảnh hoàn chỉnh với PyTorch (từ dataset đến evaluation).
* Train CNN baseline và fine-tune pretrained model (ResNet18) trên CIFAR-10.
* So sánh các optimizer và learning rate, đánh giá tác động đến chất lượng mô hình.
* Triển khai được trên cả môi trường local (CPU) và cloud GPU (Kaggle).

---

## 2. Công Việc Đã Hoàn Thành

Các việc đã làm trong tuần:

* **Thực hành tensor, autograd, debug shape** - Hoàn thành script `buoi1_tensor_autograd.py`, kiểm tra forward pass CNN, xử lý lỗi shape mismatch.
* **Xây dựng data pipeline** - Tải và xử lý CIFAR-10, tạo train/val/test split cố định, visualize batch mẫu.
* **Huấn luyện CNN baseline** - Thiết kế mô hình `SmallCNN`, train trên local CPU (10 epochs) và Kaggle GPU (15 epochs), lưu checkpoint tốt nhất.
* **Fine-tune ResNet18** - Thử nghiệm 2 chiến lược:
  - **Lần 1 (local):** freeze toàn bộ backbone → kết quả kém (~46.7% val acc).
  - **Lần 2 (Kaggle):** unfreeze toàn bộ, dùng lr thấp và scheduler → đạt 92.64% test accuracy.
* **So sánh optimizer & learning rate** - Trên Kaggle, chạy 4 cấu hình (AdamW/SGD × lr 1e-3/1e-2) với CNN, lưu bảng so sánh.
* **Đánh giá và trực quan** - Tính accuracy, macro F1, confusion matrix, vẽ learning curve cho từng mô hình.
* **Tạo báo cáo tổng hợp** - File `report.md` và log kết quả trên Kaggle; tài liệu này tổng hợp cả hai lần chạy.

Link liên quan:

* Code/PR: Kaggle Notebook (đính kèm), source local trong thư mục `https://github.com/NamKhanh2128/ai-from-scratch/tree/phase02`
* Notebook/report: Kaggle output `report.md`, các file ảnh learning curve, confusion matrix
* Dataset: CIFAR-10 (tự động tải qua `torchvision`)

---

## 3. Artifact Của Tuần

Tuần này đã tạo ra các artifact sau:

| Artifact | Mô tả | Trạng thái | Link |
|---|---|---|---|
| CNN baseline checkpoint (local) | `cnn_best.pt` – val acc 68.38% | Done | `outputs/phase2/cnn_best.pt` (local) |
| CNN baseline checkpoint (Kaggle) | `cnn_best.pt` – val acc 66.44% (AdamW) | Done | Kaggle `/kaggle/working/outputs/cnn_best.pt` |
| ResNet18 full fine-tune checkpoint (Kaggle) | `resnet_full_best.pt` – val acc ~92% | Done | Kaggle `/kaggle/working/outputs/resnet_full_best.pt` |
| Experiment comparison CSV | So sánh 4 cấu hình optimizer/lr | Done | Kaggle `/kaggle/working/reports/experiment_comparison.csv` |
| Learning curves & confusion matrices | 4 file ảnh PNG | Done | Kaggle `reports/*.png` |
| Báo cáo tổng hợp Markdown | `report.md` (Kaggle) | Done | Kaggle `report.md` |

---

## 4. Kết Quả Đo Lường

### 4.1. So sánh CNN Baseline và ResNet18 (trên Kaggle – môi trường tối ưu nhất)

| Metric | CNN Baseline | ResNet18 Fine-tune (full unfreeze) |
|---|---|---|
| Test Accuracy | 72.75% | 92.64% |
| Macro F1 | 0.7284 | 0.9264 |
| Thời gian/epoch (GPU) | ~30s | ~250s |

### 4.2. So sánh Optimizer & Learning Rate trên CNN Baseline (Kaggle)

| Cấu hình | Best Val Accuracy | Final Val Loss |
|---|---|---|
| AdamW lr=1e-3 | 66.44% | 1.036 |
| AdamW lr=1e-4 | 64.40% | 1.030 |
| **SGD lr=1e-2** | **69.00%** | **0.937** |
| SGD lr=1e-3 | 62.60% | 1.066 |

Nhận xét: SGD với momentum=0.9 và lr=0.01 cho kết quả tốt nhất trong các thử nghiệm, hội tụ ổn định và val loss thấp nhất.

### 4.3. Kết quả lần 1 (local CPU) – chỉ CNN đầy đủ

- **CNN Baseline (10 epochs):** Test accuracy 71.91%, Macro F1 71.33%.
- **ResNet18 (freeze backbone):** Val accuracy 46.70%, không khả quan.

**Lưu ý:** Sự khác biệt giữa local và Kaggle là do số epoch (10 vs 15), batch size (64 vs 128), và thiết bị (CPU vs GPU). Kaggle cho điều kiện huấn luyện tốt hơn, đặc biệt với ResNet.

---

## 5. Phân Tích Lỗi / Vấn Đề Gặp Phải

| Vấn đề | Nguyên nhân hiện tại | Ảnh hưởng | Hướng xử lý |
|---|---|---|---|
| ResNet18 lần 1 accuracy thấp (~46%) | Freeze toàn bộ backbone khi đã thay đổi stem (conv1, maxpool) khiến feature không còn phù hợp, classifier không đủ năng lực học | Mô hình không học được, thua xa CNN baseline | Chuyển sang full fine-tune với lr thấp, thêm scheduler. Kết quả lần 2 tăng vọt lên 92.64% |
| Ghi đè file history khi train nhiều model | Cùng tên file `phase2_cnn_history.json` dùng chung cho cả CNN và ResNet | Mất dữ liệu lịch sử của CNN khi so sánh | Sửa script để lưu history theo tên model riêng (đã khắc phục trên Kaggle) |
| Cảnh báo `pin_memory` trên CPU | Thiết lập `pin_memory=True` khi không có GPU | Không ảnh hưởng kết quả nhưng gây nhiễu log | Chuyển `pin_memory=False` khi chạy CPU |
| Shape mismatch khi sửa ResNet stem | Phần flatten không khớp với fc do tính sai kích thước feature map cuối | Training không chạy được | Thử lại bằng cách cho chạy forward với input giả để lấy đúng số feature (đã làm đúng ở Kaggle) |

---

## 6. Những Gì Đã Học Được

Tóm tắt kiến thức hoặc kỹ năng mới học được:

* **Transfer learning yêu cầu điều chỉnh phù hợp:** Chỉ thay head và freeze backbone không hiệu quả nếu ảnh đầu vào khác biệt nhiều về kích thước/phân phối. Cần unfreeze hoặc ít nhất cho các lớp sau học lại với lr nhỏ.
* **Cách tổ chức thí nghiệm:** Dùng seed cố định, lưu history riêng biệt, tự động hóa so sánh optimizer/LR để có kết luận định lượng.
* **Tận dụng môi trường GPU:** Việc chuyển từ local CPU lên Kaggle GPU không chỉ tăng tốc độ mà còn cho phép chạy đủ epoch và thử nghiệm nhiều hơn (15 epoch, 4 cấu hình optimizer).
* **Debug hiệu quả với shape và device:** Sử dụng cách in shape sau mỗi block và kiểm tra thiết bị trước khi train giúp phát hiện lỗi sớm.

---

## 7. Blockers / Cần Hỗ Trợ

* Hiện tại không có blocker lớn.

---

## 8. Kế Hoạch Tuần Sau

Các mục tiêu chính cho tuần sau:

* Nghiên cứu các kỹ thuật augmentation nâng cao (CutMix, MixUp) để cải thiện CNN baseline.
* Thử nghiệm kiến trúc MobileNetV2 cho edge deployment.

Output dự kiến tuần sau:

* Code augmentation và kết quả so sánh.
* Checkpoint MobileNetV2 fine-tune trên CIFAR-10.

---

## 9. Self-Assessment

Intern tự đánh giá tiến độ tuần này:

**Mức độ hoàn thành:** 95%

**Chất lượng output:** Tốt

**Mức độ tự chủ:** Có thể tự xử lý phần lớn (từ việc debug local đến triển khai Kaggle, tự sửa lỗi freeze và cải thiện kết quả) với AI tool

**Nhận xét ngắn:** Tuần này đã hoàn thành toàn bộ mục tiêu Phase 2, bao gồm cả việc chuyển đổi môi trường và tối ưu mô hình. Kết quả cuối cùng vượt mong đợi nhờ điều chỉnh chiến lược fine-tune đúng đắn. 