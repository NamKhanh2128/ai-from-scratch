# Weekly Report - Week 1

**Intern:** Nguyễn Đức Nam Khánh
**Thời gian:** 14/05/2026 - 17/05/2026
**Phase hiện tại:** 0 - Setup & Engineering Workflow
**Mentor/Reviewer:** Nguyễn Bảo Anh

---

## 1. Mục Tiêu Tuần Này

Các mục tiêu đã đặt ra cho tuần này:

Hoàn thành các phần công việc trong phrase 0
---

## 2. Công Việc Đã Hoàn Thành

Các việc đã làm trong tuần:

- Dev Environment – Cài đặt thành công Python 3.12, Node.js 22, Rust, Julia, PyTorch; tạo môi trường ảo bằng uv.

- Git & Collaboration – Tạo repo ai-from-scratch, viết .gitignore, thực hành branch/merge, push code lên GitHub qua SSH.

- GPU Setup & Cloud – Chạy Google Colab với GPU T4 miễn phí; benchmark CPU vs GPU và ước lượng mô hình theo quy tắc fp16.

- APIs & Keys – Đăng ký Gemini API key, lưu an toàn trong .env, gọi API thành công bằng Python SDK và raw HTTP.

- Jupyter Notebooks – Cài JupyterLab, thành thạo code/markdown cells, magic commands, hiển thị DataFrame và đồ thị.

- Python Environments – Tạo môi trường ảo bằng uv, viết pyproject.toml, tạo lockfile, thực hành cài global và gỡ bỏ.

- Docker for AI – Đã kiểm tra Docker có sẵn, đang viết Dockerfile và chuẩn bị build image.

Link liên quan:

* Code/PR: https://github.com/NamKhanh2128/ai-from-scratch
* Demo: https://colab.research.google.com/drive/1O3_WxitFMQSV79_wVLAYuMUid-vhwyeg#scrollTo=l3ZbO2UJuSky
* Notebook/report: [link]
* Dataset/config: [link]

---

## 3. Artifact Của Tuần

Tuần này đã tạo ra các artifact sau:

https://github.com/NamKhanh2128/ai-from-scratch/blob/main/Screenshot%202026-05-17%20210424.png

---

## 4. Kết Quả Đo Lường

https://github.com/NamKhanh2128/ai-from-scratch/blob/main/Screenshot%202026-05-17%20205827.png
Ghi chú:
- Các phép đo GPU thực hiện trên Google Colab (free tier) với GPU Tesla T4.
- Thời gian gọi API có thể thay đổi tùy theo tải của Google.
- List comprehension và numpy benchmark thực hiện trong môi trường ảo WSL (CPU Intel/AMD).

---

## 5. Phân Tích Lỗi / Vấn Đề Gặp Phải

https://github.com/NamKhanh2128/ai-from-scratch/blob/main/Screenshot%202026-05-17%20210604.png

---

## 6. Những Gì Đã Học Được

Tóm tắt kiến thức hoặc kỹ năng mới học được:

- Thiết lập môi trường phát triển AI đa ngôn ngữ (Python, Node.js, Rust, Julia) với các công cụ quản lý gói hiện đại (uv, pnpm, cargo) và môi trường ảo độc lập cho từng dự án.

- Sử dụng Git để quản lý phiên bản (branch, merge, commit, push) và GitHub qua SSH; lưu trữ an toàn API key bằng file .env và .gitignore.

- Làm chủ Jupyter Notebooks như công cụ thí nghiệm AI chính: code cells, markdown, magic commands (%timeit, %%time), hiển thị trực quan (DataFrame, đồ thị) và xử lý kernel.

- Gọi mô hình ngôn ngữ lớn qua API (Gemini) bằng cả Python SDK và raw HTTP, hiểu cơ chế xác thực, rate limit, và cách debug lỗi API thường gặp.

- Sử dụng GPU miễn phí trên Google Colab (T4) để benchmark và huấn luyện, cùng các nguyên tắc ước lượng kích thước mô hình theo VRAM (quy tắc fp16).

- Đóng gói môi trường AI bằng Docker (Dockerfile, volume mount, Docker Compose) đảm bảo tính nhất quán khi triển khai.

---

## 7. Blockers / Cần Hỗ Trợ

Chưa có

---

## 8. Kế Hoạch Tuần Sau

- Hoàn thành toàn bộ bài học Phase 01 (Math Foundations).

- Thực hành cài đặt các phép toán đại số tuyến tính, giải tích, xác suất và tối ưu bằng Python.

- Gắn kết các khái niệm toán học với ứng dụng trong AI (gradient descent, backpropagation,...).

Output dự kiến:

- Bộ notebook Jupyter cho từng chủ đề: Linear Algebra, Calculus, Probability, Optimization.

- Script Python triển khai các thuật toán nền tảng (phân rã ma trận, gradient descent,...).

- Tài liệu tóm tắt công thức và sơ đồ liên kết với các phase sau (Neural Networks, Deep Learning).


---

## 9. Self-Assessment

Mức độ hoàn thành: 85%
(Cơ bản hoàn thành 7/12 bài học; bài Docker for AI đang thực hiện dở, còn 4 bài cuối Phase 00 chưa làm)

Chất lượng output: Đạt
(Các công cụ được cài đặt đúng, API hoạt động, notebook chạy được; tuy nhiên còn một số lỗi nhỏ như quản lý file môi trường, chưa tối ưu Dockerfile)

Mức độ tự chủ: Có thể tự xử lý phần lớn với sự hỗ trợ của AI

Nhận xét ngắn:
Đã làm chủ được môi trường phát triển AI cơ bản, biết cách debug các lỗi thường gặp (PATH, virtual environment, API key). Hoàn thành tốt các bài tập về Git, Jupyter, API và Python Environments. Cần tập trung hoàn thiện Docker và các bài còn lại trong Phase 00 trước khi bước vào Math Foundations.

