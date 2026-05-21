# Weekly Report - Week 2

**Intern:** Nguyễn Đức Nam Khánh

**Thời gian:** 18/05/2026 - 21/05/206

**Phase hiện tại:** Phase 01 - ML Foundation & Evaluation

**Mentor/Reviewer:** Nguyễn Bảo Anh

---

## 1. Mục Tiêu Tuần Này

Các mục tiêu đã đặt ra cho tuần này:

- Xây dựng pipeline huấn luyện và đánh giá mô hình phân loại nhị phân hoàn chỉnh.
- Thực hành viết script tổng quát với config YAML, hỗ trợ grid search đơn giản.
- Tạo báo cáo tổng kết, trực quan hóa confusion matrix và so sánh metric giữa các mô hình.


---

## 2. Công Việc Đã Hoàn Thành

Các việc đã làm trong tuần:

- **Dataset & Split**
  - Load Breast Cancer Wisconsin (569 mẫu, 30 features)
  - Chia train/val/test (65/15/20) stratified, seed = 42

- **Logistic Regression**
- Áp dụng `StandardScaler`
- Thử các giá trị `C`: `0.1, 1.0, 10.0`
- Chọn model tốt nhất theo validation F1  
→ **- Best: `C = 0.1`, **Val F1 = 1.0000****

**Test:**
- F1: 0.9748  
- Accuracy: 0.9677  
- Precision: 0.9667  
- Recall: 0.9831  
- Confusion Matrix: `[[32, 2], [1, 58]]`

---

### Decision Tree
- Thử `max_depth`: 3, 5, None
- Best: `max_depth = 3`, **Val F1 = 0.9600**

**Test:**
- F1: 0.9500  
- Accuracy: 0.9355  
- Precision: 0.9344  
- Recall: 0.9661  
- Confusion Matrix: `[[30, 4], [2, 57]]`

---

### Random Forest
- Thử `n_estimators`: 50, 100
- Best: `n_estimators = 50`, **Val F1 = 0.9630**

**Test:**
- F1: 0.9661  
- Accuracy: 0.9570  
- Precision: 0.9661  
- Recall: 0.9661  
- Confusion Matrix: `[[32, 2], [2, 57]]`

---

- **Script tổng quát**
  - `train_phase1.py`: nhận config YAML, grid search, lưu model & scaler
  - `evaluate_phase1.py`: tính metrics, vẽ confusion matrix, lưu CSV
  - `compare_models.py`: vẽ biểu đồ so sánh các model

- **Báo cáo**
  - `phase1_report.md`: mô tả bài toán, phương pháp, kết quả, phân tích lỗi

- **Reproducibility**
  - Xóa toàn bộ output và chạy lại pipeline
  - Kết quả khớp nhờ seed cố định

---

Link liên quan:

* Code/PR: https://github.com/NamKhanh2128/ai-from-scratch/tree/phase01
* Demo: `phase1_ml/reports/phase1_report.md`
* Notebook/report: `phase1_ml/reports/phase1_report.md`
* Dataset/config: Breast Cancer Wisconsin (scikit-learn), `configs/*.yaml`

---

## 3. Artifact Của Tuần

Tuần này đã tạo ra các artifact sau:

| Artifact | Mô tả | Trạng thái | Link |
|----------|------|-----------|------|
| train_phase1.py | Script huấn luyện tổng quát | Done | phase1_ml/scripts/train_phase1.py |
| evaluate_phase1.py | Script đánh giá & lưu metrics | Done | phase1_ml/scripts/evaluate_phase1.py |
| compare_models.py | So sánh metric giữa các model | Done | phase1_ml/scripts/compare_models.py |
| Config YAML | Cấu hình các mô hình | Done | phase1_ml/configs/ |
| Models & scaler | File pickle | Done | phase1_ml/outputs/ |
| phase1_results.csv | Tổng hợp metrics | Done | phase1_ml/reports/phase1_results.csv |
| phase1_report.md | Báo cáo Phase 1 | Done | phase1_ml/reports/phase1_report.md |
| Figures | Confusion matrix & chart | Done | phase1_ml/reports/figures/ |

---

---

## 4. Kết Quả Đo Lường

| Metric | Baseline | Logistic Regression | Decision Tree | Random Forest | Nhận xét |
|--------|--------|---------------------|--------------|--------------|---------|
| Accuracy | 0.500 | **0.9677** | 0.9355 | 0.9570 | LogReg tốt nhất |
| Precision | 0.500 | **0.9667** | 0.9344 | 0.9661 | LogReg & RF cao |
| Recall | 0.500 | **0.9831** | 0.9661 | 0.9661 | LogReg cao nhất |
| F1-score | 0.500 | **0.9748** | 0.9500 | 0.9661 | LogReg vượt trội |

**Dataset:** Breast Cancer Wisconsin (scikit-learn)

---

### Reproduce command

python phase1_ml/scripts/train_phase1.py --config phase1_ml/configs/phase1_logreg.yaml
python phase1_ml/scripts/train_phase1.py --config phase1_ml/configs/phase1_tree.yaml
python phase1_ml/scripts/train_phase1.py --config phase1_ml/configs/phase1_rf.yaml

python phase1_ml/scripts/evaluate_phase1.py --model-path ... --config ...

--- 
### Config

- LogisticRegression(C=0.1, max_iter=2000)
- DecisionTree(max_depth=3)
- RandomForest(n_estimators=50)
- random_state=42
- test_size=0.2
- val_size=0.15
---

## 5. Phân Tích Lỗi / Vấn Đề

| Vấn đề | Nguyên nhân | Ảnh hưởng | Hướng xử lý |
|--------|------------|----------|------------|
| LogisticRegression not subscriptable | Load model sai (treat như dict) | Không evaluate được | Sửa lại load pickle đúng |
| Thiếu scaler | Train chưa lưu scaler | Dự đoán sai | Lưu scaler khi `use_scaler=True` |
| CSV bị mất | File bị xóa trước reproduce | Không so sánh được | Auto create nếu chưa tồn tại |
| Lỗi terminal | Copy command sai format | Chạy lỗi | Chạy từng lệnh |

**Case lỗi tiêu biểu:**

- **Input:** Chạy `evaluate_phase1.py` nhưng thiếu `--scaler-path`
- **Output lỗi:** `TypeError: 'LogisticRegression' object is not subscriptable`
- **Fix:** Thêm argument `--scaler-path` + load đúng scaler từ file

---

## 6. Những Gì Đã Học Được

- Xây dựng pipeline ML end-to-end (data → train → eval → report)
- Sử dụng YAML config để quản lý experiment
- Viết script reusable (train / evaluate / compare)
- Debug lỗi Python liên quan đến:
  - File I/O
  - Model serialization (pickle)
  - Đồng bộ train & evaluate
- Visualization:
  - Confusion matrix
  - So sánh metric bằng bar chart

---

## 7. Blockers / Cần Hỗ Trợ

- Không có blocker lớn
- Các lỗi nhỏ đã tự debug và xử lý

---


## 8. Kế Hoạch Tuần Sau

- Bắt đầu **Phase 2: Neural Networks**
- Xây dựng mạng fully-connected bằng NumPy

Output dự kiến tuần sau:

- Script / Notebook neural network from scratch
- Báo cáo so sánh hiệu năng
---

## 9. Self-Assessment

- **Mức độ hoàn thành:** 100%
- **Chất lượng output:** Tốt
- **Mức độ tự chủ:** Tự chủ ổn định

**Nhận xét:**

Hoàn thành đầy đủ Phase 1, pipeline ML hoạt động ổn định và reproducible.  
Đã nắm được sơ bộ pipeline huấn luyện, đánh giá và so sánh mô hình.  
Sẵn sàng chuyển sang Neural Network và áp dụng kiến thức sâu hơn.

---
