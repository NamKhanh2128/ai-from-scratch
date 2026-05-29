# scripts/buoi1_tensor_autograd.py
import torch
import torch.nn as nn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tạo tensor, kiểm tra shape, dtype, device
    a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print("Tensor a:\n", a)
    print("Shape:", a.shape, "Dtype:", a.dtype, "Device:", a.device)
    
    # Chuyển sang GPU nếu có
    a = a.to(device)
    b = torch.tensor([[0.5], [1.0]], device=device, requires_grad=True)
    
    # 2. Phép toán và backward
    c = a @ b                 # matrix multiply -> shape (2,1)
    loss = c.sum()            # scalar loss
    print("Loss:", loss.item())
    
    # Backward
    loss.backward()
    print("Gradient of a:\n", a.grad)
    print("Gradient of b:\n", b.grad)
    
    # 3. Mô phỏng một bước optimizer (zero_grad, step)
    # Tạo một tham số mô phỏng
    w = torch.tensor([2.0], requires_grad=True, device=device)
    x = torch.tensor([3.0], device=device)
    target = torch.tensor([12.0], device=device)
    
    # Forward
    y = w * x
    loss = (y - target).pow(2).mean()   # MSE
    print(f"Initial loss: {loss.item():.4f}, w: {w.item():.4f}")
    
    # Backward
    loss.backward()
    print("Gradient of w:", w.grad.item())
    
    # Cập nhật thủ công (giống SGD)
    lr = 0.1
    with torch.no_grad():
        w -= lr * w.grad
    # Xóa gradient tích lũy
    w.grad.zero_()
    print(f"After update: w: {w.item():.4f} (expected about 4.0)")
    
    # 4. Kiểm tra shape qua một lớp mạng đơn giản
    sample_input = torch.randn(1, 1, 28, 28, device=device)  # batch=1, channel=1, 28x28
    conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1).to(device)
    out = conv(sample_input)
    print("Input shape:", sample_input.shape)
    print("Output shape after Conv2d:", out.shape)
    
    # Tạo lỗi shape mismatch có kiểm soát (giải thích)
    # Ví dụ: flatten rồi cho vào Linear với input size không khớp
    flat = out.view(1, -1)  # shape (1, 4*28*28)
    try:
        linear = nn.Linear(4*28*28 + 5, 10).to(device)  # cố tình sai
        _ = linear(flat)
    except RuntimeError as e:
        print("Shape mismatch error:", e)
    print("Sửa: input features của Linear phải khớp với số phần tử sau flatten")

if __name__ == "__main__":
    main()
