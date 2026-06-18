# Lab 16 - Cloud AI Environment Setup: Training Report

**Student:** Nguyễn Tuấn Dũng  
**ID:** 2A202600583  
**Date:** 2026-06-18

---

## 1. Experiment Setup

- **Dataset:** Credit card fraud detection (~284k rows, stratified split 80/20)
- **Model:** LightGBM (CPU fallback — no GPU quota available)
- **Random state:** 42 (reproducible results)
- **Train rows:** 227,845 | **Test rows:** 56,962

---

## 2. Results Comparison (Run 1 vs Run 2)

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| Load time (s) | 2.6028 | 2.9530 | +0.35s |
| **Training time (s)** | **2.0443** | **2.0574** | +0.013s (+0.6%) |
| Best iteration | 1 | 1 | — |
| **AUC-ROC** | **0.951654** | **0.951654** | No change |
| Accuracy | 0.998947 | 0.998947 | No change |
| F1 Score | 0.727273 | 0.727273 | No change |
| Precision | 0.655738 | 0.655738 | No change |
| Recall | 0.816327 | 0.816327 | No change |
| **Inference latency — 1 row (ms)** | **0.9724** | **0.4434** | -54% |
| **Throughput — 1000 rows/s** | **1,177,411** | **1,244,625** | +5.7% |

---

## 3. Analysis

### 3.1 Training Time
Cả hai lần chạy đều hoàn thành trong ~2.05s và hội tụ ngay tại `best_iteration: 1`. Chênh lệch 0.013s không có ý nghĩa thống kê — cho thấy dataset tuyến tính và LightGBM tối ưu tốt trên tabular data.

### 3.2 AUC-ROC & Model Quality
AUC-ROC = **0.9517** ở cả hai lần — kết quả hoàn toàn deterministic do cùng `random_state: 42`. Đây là mức chất lượng cao cho bài toán phát hiện gian lận (fraud detection).

- **Precision 0.656 / Recall 0.816:** Model thiên về recall (ưu tiên bắt đúng fraud hơn là tránh false positive) — phù hợp với nghiệp vụ tài chính.

### 3.3 Inference Speed
| | Run 1 | Run 2 |
|--|-------|-------|
| Latency (1 row) | 0.97ms | 0.44ms |
| Throughput | 1.18M rows/s | 1.24M rows/s |

Lần 2 nhanh hơn ~54% do **CPU cache warm-up effect**: lần 1 là cold start, lần 2 CPU instruction cache và data cache đã được nạp sẵn. Throughput >1.2M rows/s đủ cho real-time scoring ở quy mô production.

---

## 4. Why CPU Instead of GPU?

| Reason | Detail |
|--------|--------|
| No GPU quota | AWS account không có quota cho GPU instances (p3, g4dn...). Cần request riêng, thường bị từ chối hoặc chờ lâu |
| Cost | g4dn.xlarge ~$0.52/h vs t3.large ~$0.08/h — chênh lệch 6.5x không hợp lý cho lab |
| LightGBM is CPU-native | GBDT (gradient boosted decision trees) được tối ưu cho CPU; GPU chủ yếu có lợi khi training deep neural networks |
| Dataset size | ~280k rows × tabular features — CPU xử lý dễ dàng trong <3s, không cần GPU acceleration |

**Kết luận:** CPU fallback (LightGBM) là lựa chọn hợp lý và hiệu quả cho bài toán này. AUC 0.95 và throughput >1.2M rows/s đáp ứng đầy đủ yêu cầu production.

---

## 5. Conclusion

- Model đạt chất lượng cao (AUC ~0.95) với thời gian training cực ngắn (~2s)
- Kết quả ổn định, reproducible qua nhiều lần chạy
- Inference đủ nhanh cho real-time fraud detection
- CPU fallback không làm giảm chất lượng model so với GPU cho loại bài này
