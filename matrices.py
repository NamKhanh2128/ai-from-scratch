from vectors import Vector  # dùng lại Vector của bài trước

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.rows > 0 else 0
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    # ── Các phép toán cơ bản ──
    def __add__(self, other):
        """Cộng hai ma trận cùng shape."""
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        """Trừ hai ma trận cùng shape."""
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        """Nhân vô hướng."""
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        """Nhân từng phần tử (Hadamard product) – yêu cầu cùng shape."""
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        """Nhân ma trận (m x n) @ (n x p) -> (m x p)."""
        if self.cols != other.rows:
            raise ValueError(f"Shape mismatch: {self.shape} and {other.shape}")
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    # ── Biến đổi ──
    def transpose(self):
        """Chuyển vị."""
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    # ── Định thức (đệ quy, chấp nhận ma trận vuông) ──
    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinant only defined for square matrices.")
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    # ── Nghịch đảo 2x2 ──
    def inverse_2x2(self):
        if self.shape != (2, 2):
            raise ValueError("inverse_2x2 only works for 2x2 matrices.")
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse.")
        return Matrix([
            [ self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det,  self.data[0][0] / det]
        ])

    # ── Ma trận đơn vị (identity) ──
    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    # ── Hỗ trợ toán tử @ (dùng matmul) ──
    def __matmul__(self, other):
        # Nếu other là Vector, dùng cách nhân cũ
        if isinstance(other, Vector):
            return Vector([
                sum(self.data[i][j] * other.components[j] for j in range(self.cols))
                for i in range(self.rows)
            ])
        return self.matmul(other)