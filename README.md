# Khảo sát Gauss–Seidel và mức độ chéo trội

## Chạy chương trình

Từ thư mục `BTL`, dùng môi trường hiện có:

```powershell
.\.venv\Scripts\python.exe .\src\program.py
```

Mặc định giữ các kích thước hiện tại `n = 3, 16, 24, 50, 100`, các mức `q = 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0, 1.05, 1.2, 1.8` và 20 seed từ 42 đến 61. Bên cạnh bộ đặc, thêm các bộ thưa có đúng `k = 2, 4, 8` phần tử ngoài đường chéo khác 0 trên mỗi hàng. Bỏ qua `k >= n-1`: mức bằng `n-1` đã có trong bộ đặc, mức lớn hơn không khả thi. Vì vậy `n=3` chỉ chạy bộ đặc. Tổng mặc định là **3.740 trường hợp**, trong đó bộ đặc có 1.100 trường hợp. Ngưỡng dừng `1e-8`, tối đa 10.000 vòng lặp.

Chạy nhanh hoặc đổi bộ thực nghiệm:

```powershell
.\.venv\Scripts\python.exe .\src\program.py --sizes 3 10 --q-values 0.5 0.99 1.2 --seeds 42 43 --output-dir outputs\quick
.\.venv\Scripts\python.exe .\src\program.py --help
```

Chỉ chạy bộ đặc như trước, hoặc tùy chọn mức thưa:

```powershell
.\.venv\Scripts\python.exe .\src\program.py --dense-only
.\.venv\Scripts\python.exe .\src\program.py --neighbors 1 2 4
```

Các mức `--neighbors` bổ sung vào bộ đặc, không thay thế nó. Các hàm `prepare_experiments()` và `run_experiments_on_size()` khi gọi riêng vẫn mặc định chỉ dùng bộ đặc; truyền `neighbors=[2,4,8]` để thêm các bộ thưa.

Các kích thước lớn như 512 hoặc 1331 vẫn có thể truyền qua `--sizes`, nhưng phép tính đầy đủ giá trị riêng của ma trận lặp dày đặc sẽ tốn thêm thời gian và bộ nhớ.

## Tổ chức mã nguồn

| File trong `src` | Trách nhiệm |
| --- | --- |
| `prepare.py` | Cấu hình bộ mặc định, sinh các bộ đặc/thưa theo q và số tương tác mỗi hàng, tính q thực tế, chuẩn bị A, b, x_exact, x0. `prepare_experiments()` lần lượt trả từng `ExperimentCase`. |
| `solve.py` | Bộ giải `solve_gauss_seidel()`, sai số chuẩn vô cùng, ma trận lặp và bán kính phổ. Có thể import để giải một hệ riêng mà không nạp Matplotlib. |
| `visualization.py` | Hàm `summarize()` tổng hợp riêng từng bộ và `save_plots()` lưu các biểu đồ theo bộ cùng biểu đồ đối chiếu độ thưa. |
| `program.py` | Nhận tham số dòng lệnh, nối phần chuẩn bị với bộ giải, gắn thông tin thực nghiệm, xuất CSV/JSON và gọi phần vẽ. |
| `utils.py` | Kiểu dữ liệu `Result` và các hàm kiểm tra mảng dùng chung. |
| `test_program.py` | Kiểm tra bộ giải, sinh dữ liệu và pipeline. |

Luồng gọi: `program.py` -> `prepare.prepare_experiments()` -> `solve.solve_gauss_seidel()` -> lưu CSV -> `visualization.summarize()` / `visualization.save_plots()`.

Ví dụ sử dụng riêng bộ chuẩn bị và bộ giải (chạy từ thư mục `src`):

```python
from prepare import prepare_experiments
from solve import solve_gauss_seidel

cases = prepare_experiments(10, q_values=[0.9], seeds=[42], neighbors=[2])
case = next(c for c in cases if c.family == "sparse_k2")
result = solve_gauss_seidel(
    case.A, case.b, case.x0, x_exact=case.x_exact,
)
print(result.status, result.k, result.final_error)
```

## Pipeline

1. Với mỗi cặp `(n, seed)`, sinh ma trận nền bằng `default_rng(seed).uniform(-1, 1, (n,n))`, đặt đường chéo nền bằng 0.
2. Với bộ thưa, dùng một luồng sinh số độc lập `SeedSequence([seed,n,9173])` để hoán vị các cột ngoài đường chéo trên từng hàng. Lấy k vị trí đầu, giữ nguyên hệ số từ ma trận nền đặc ở các vị trí đó. Các tập vị trí lồng nhau khi tăng k; việc thêm hoặc đổi thứ tự mức k không làm đổi bộ đặc.
3. Giữ nguyên cả vị trí khác 0 và các hệ số ngoài đường chéo trong mỗi bộ khi thay đổi q. Đặt `a_ii = sum(j != i, abs(a_ij)) / q`; mọi hàng phải có ít nhất một phần tử ngoài đường chéo khác 0. Tính lại `q_actual` để đối chiếu.
4. Chọn `x_exact = ones(n)`, `b = A @ x_exact`, `x0 = zeros(n)`.
5. Giải bằng GS theo thứ tự hàng 0 đến n−1. Dùng ngay các thành phần mới ở j < i, dùng thành phần cũ ở j > i.
6. Dừng khi `norm(b - A @ x, inf) / max(1, norm(b, inf)) <= tol`. Nghiệm đúng chỉ dùng đo sai số, không tham gia điều kiện dừng.
7. Với `A = D + L + U`, tính `B = solve(D + L, -U)` và `rho = max(abs(eigvals(B)))`.
8. Lưu kết quả từng trường hợp, lịch sử từ vòng 0, cấu hình, thống kê và ba biểu đồ.

## Định nghĩa độ thưa

`neighbors` là số phần tử ngoài đường chéo khác 0 trên mỗi hàng. `density = k/(n-1)` và `sparsity = 1-density` chỉ tính trên các vị trí ngoài đường chéo. Ví dụ n=100, k=2 có mật độ 2/99 và độ thưa 97/99. Đường chéo được tính lại theo q và luôn khác 0. Ma trận không được ép đối xứng; mỗi hàng chọn các cột độc lập.

Trong một cặp (n, seed), các mức k dùng tập vị trí lồng nhau và chia sẻ hệ số tại các vị trí chung. Mỗi (n, seed, family) có một phần ngoài đường chéo cố định khi thay q. Khi tăng n ở k cố định, số tương tác trực tiếp trên mỗi hàng giữ nguyên.

Chương trình vẫn lưu ma trận bằng NumPy ndarray và dùng bộ giải GS hiện tại. Độ thưa ở đây là biến thực nghiệm về cấu trúc ma trận, chưa phải triển khai bộ giải tối ưu bộ nhớ/thời gian bằng định dạng sparse.

## Kết quả

Mặc định xuất vào thư mục `outputs` (không phụ thuộc thư mục gọi lệnh). Chạy lại cùng thư mục sẽ cập nhật các file này; dùng `--output-dir` khác nếu cần giữ riêng mỗi lần chạy. File `src/results.csv` cũ được giữ nguyên.

- `results.csv`: các cột cũ cùng `family`, `neighbors`, `density`, `sparsity`. `family` là `dense`, `sparse_k2`, `sparse_k4` hoặc `sparse_k8`. Khóa của một trường hợp là `(n, seed, family, q_target)`.
- `convergence_history.csv`: n, seed, q_target, iteration, residual, error. Có thêm `family`, `neighbors`, `density`, `sparsity` để phân biệt lịch sử từng bộ. Error là chuẩn vô cùng của sai số nghiệm; vòng 0 là nghiệm ban đầu.
- `summary.csv`: tổng hợp riêng theo `(n, q_target, family)`, không gộp bộ đặc và các mức k. Ghi số trường hợp của từng trạng thái, tỷ lệ đạt ngưỡng dừng, trung vị và tứ phân vị số vòng lặp của các trường hợp đạt ngưỡng, thống kê bán kính phổ.
- `config.json`: cấu hình sinh dữ liệu, chuẩn và tiêu chuẩn dừng, phiên bản Python và các thư viện.
- `q_vs_iterations.png`: trung vị và khoảng tứ phân vị số vòng lặp. Trục Y của biểu đồ trên dùng thang logarit ở vùng lớn hơn 1 để các đường có số vòng lặp nhỏ không bị nén bởi giá trị lớn; vùng 0–1 vẫn tuyến tính (symlog) để biểu diễn được 0 vòng lặp. Biểu đồ tỷ lệ đạt ngưỡng phía dưới giữ nguyên thang tuyến tính. Gọi riêng `save_iteration_plot(summary, output_dir)` để vẽ lại ảnh này từ bảng thống kê.
- `q_vs_spectral_radius.png`: trung vị và khoảng tứ phân vị bán kính phổ, kèm đường rho = 1.
- `error_vs_iterations.png`: đồ thị sai số trên trục tung logarit; một seed cố định, tối đa 5 mức q cho mỗi kích thước. Giới hạn hiển thị mặc định từ `1e-12` đến `10`. Khi sai số vượt 10, đường vẽ dừng tại dấu tam giác kèm chú thích; không kéo dài thành đường ngang giả. Giá trị dưới `1e-12` chỉ được kẹp khi vẽ. CSV giữ nguyên toàn bộ dữ liệu. Có thể gọi riêng `save_error_plot(history_path, output_dir, example_seed, error_ceiling=10.0, error_floor=1e-12)` trong `visualization.py` để vẽ lại từ lịch sử có sẵn hoặc đổi giới hạn mà không chạy lại bộ giải.

Ba trạng thái: `converged` (đạt ngưỡng phần dư), `max_iterations` (hết giới hạn vòng lặp), `numerical_error` (xuất hiện số không hữu hạn). Không dùng riêng việc hết vòng lặp để kết luận phân kỳ. Giá trị NaN trong dữ liệu lỗi số học không phải sai số bằng 0.

`q < 1` là điều kiện đủ. `q >= 1` không tự động có nghĩa phân kỳ. Điều kiện hội tụ với mọi nghiệm ban đầu là `rho(B) < 1`; các giá trị rất gần 1 có thể chịu ảnh hưởng của làm tròn số thực. Không mặc định quan hệ đơn điệu giữa q và số vòng lặp cho mọi ma trận.

Các ảnh ở ngay `outputs/` vẫn biểu diễn **bộ đặc** như trước. Mỗi bộ thưa có thư mục riêng `outputs/sparse_k2/`, `outputs/sparse_k4/`, `outputs/sparse_k8/` với cùng ba tên ảnh. Các thư mục này được tạo nếu cấu hình có trường hợp tương ứng.

Ảnh `outputs/sparsity_comparison_n<N>.png` đối chiếu các bộ tại cùng kích thước N: số vòng lặp của các lần đạt ngưỡng (symlog), bán kính phổ và tỷ lệ đạt ngưỡng. Mỗi đường là một bộ; chú giải ghi k và mật độ thực tế. Không tạo ảnh đối chiếu nếu kích thước đó chỉ có bộ đặc. CSV đặc cũ vẫn có thể đọc và vẽ bằng các hàm trong `visualization.py`.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src -p test_program.py -v
```

Kiểm tra bao gồm cập nhật GS sau một vòng, đầu vào nguyên, chuẩn vô cùng, nghiệm ban đầu đạt ngưỡng, rho = q² cho ma trận 2 × 2 chuẩn, trường hợp q > 1 vẫn hội tụ, lỗi số học, tính tái lập và xuất dữ liệu/biểu đồ.

Các kiểm tra bổ sung xác nhận đúng k phần tử khác 0 mỗi hàng, vị trí lồng nhau, q giữ đúng, tái lập khi đổi thứ tự k, bộ đặc không đổi, thống kê không trộn bộ và xuất đủ biểu đồ đối chiếu.
