# Khảo sát Gauss–Seidel và mức độ chéo trội

## Chạy chương trình

Từ thư mục `BTL`, chỉ chạy chương trình, không truyền argument:

```powershell
.\.venv\Scripts\python.exe .\src\program.py
```

Chương trình chạy tuần tự ba bước:

1. `q-variants`: chạy mọi mức q đã cấu hình và lưu vào `output/q-variants`.
2. `non-diag-dominance`: chạy lại các mức q>=1 đã cấu hình và lưu vào `output/non-diag-dominance`.
3. Đọc dữ liệu hai bộ, kiểm tra các trường hợp trùng nhau rồi tạo đúng **3 hình báo cáo** cùng bảng số liệu gọn trong `output/final`.

Các tham số dòng lệnh trước đây đã bỏ hoàn toàn. Nếu truyền thêm argument, chương trình báo lỗi trước khi chạy. Muốn đổi bộ khảo sát, sửa `DEFAULT_SIZES`, `DEFAULT_Q_VALUES`, `DEFAULT_SEEDS`, `DEFAULT_NEIGHBORS` ở đầu `src/prepare.py`; sửa `DEFAULT_TOL`, `DEFAULT_MAX_ITERATION`, `DEFAULT_VERBOSE` ở đầu `src/program.py` để đổi ngưỡng dừng, giới hạn vòng lặp hoặc bật log từng trường hợp.

Với cấu hình hiện tại, bước đầu có **3.740 trường hợp** ở 11 mức q; bước hai có **1.360 trường hợp**, với q thuộc `{1.0,1.05,1.2,1.8}`. Cả hai dùng kích thước `{3,16,24,50,100}`, 20 seed từ 42 đến 61, các mức k `{2,4,8}` và bộ đặc. Bỏ qua k>=n-1 để không chạy trùng bộ đặc hoặc chọn mức không khả thi. Ngưỡng dừng `1e-8`, tối đa 10.000 vòng.

Tổng số lần thực thi là 5.100, nhưng chỉ có **3.740 trường hợp duy nhất**, vì bước hai là tập con q>=1 của bước đầu. Phần tổng hợp kiểm tra các kết quả trùng khớp, không đếm chúng thành các mẫu độc lập. Hai bộ khác cấu hình hoặc không khớp kết quả sẽ bị từ chối tổng hợp.

q=1 là biên không chéo trội nghiêm ngặt, khác với q>1. Việc chọn trường hợp dùng q thiết kế trong cấu hình để tránh phân loại sai vì làm tròn của q_actual. q>=1 không tự động có nghĩa phân kỳ.

```text
output/
  q-variants/             dữ liệu và biểu đồ chi tiết của mọi q
  non-diag-dominance/     dữ liệu và biểu đồ chi tiết của q>=1
  final/
    01_q_vs_iterations.png
    02_error_vs_iterations.png
    03_non_diag_dominance.png
    q_summary.csv
    convergence_by_regime.csv
    non_diagonal_examples.csv
    provenance.json
    HUONG-DAN.txt
```

Đường dẫn được tính theo thư mục dự án, không phụ thuộc thư mục gọi lệnh. Dữ liệu cũ trong `outputs/` vẫn giữ nguyên. Chạy lại chương trình sẽ cập nhật các file trong `output/`.

## Bộ hình cuối dùng trong báo cáo

- `01_q_vs_iterations.png`: câu hỏi về ảnh hưởng của mức độ chéo trội đến tốc độ. Hai ô là trung vị/khoảng tứ phân vị số vòng của các lần đạt ngưỡng và tỷ lệ đạt ngưỡng; mỗi đường là một bộ đặc/thưa.
- `02_error_vs_iterations.png`: lịch sử sai số của một seed và một bộ cố định, dùng cùng phần ngoài đường chéo khi thay q. Giới hạn hiển thị [1e-12,10], giữ chú thích và dấu tam giác khi vượt ngưỡng; không thay đổi dữ liệu gốc.
- `03_non_diag_dominance.png`: câu hỏi về hội tụ khi q>=1. Hai ô là trung vị/khoảng tứ phân vị bán kính phổ và tỷ lệ đạt ngưỡng. Có đường rho=1 để đối chiếu, nhưng trung vị rho<1 không bảo đảm mọi seed đều có rho<1.

Ba hình được xuất PNG 300 dpi. Phần tổng hợp dùng kích thước lớn nhất có trong cả hai bộ (hiện tại n=100), ưu tiên bộ sparse_k2 cho lịch sử sai số và seed đầu tiên trong cấu hình (hiện tại 42). Các q minh họa được chọn gần 0.1, 0.9, 1.2, 1.8 trong danh sách thực có; không chọn lại seed theo kết quả đẹp.

`q_summary.csv` giữ thống kê tất cả q và bộ ma trận tại kích thước được chọn (hiện tại 44 dòng). `convergence_by_regime.csv` đếm theo q<1, q=1, q>1 trên các trường hợp duy nhất của toàn bộ kích thước (hiện tại 12 dòng). `non_diagonal_examples.csv` giữ tối đa sáu ví dụ đo thực tế, ưu tiên q>1, gồm trường hợp đạt và không đạt ngưỡng nếu có. `provenance.json` ghi nguồn, số lần thực thi/số trường hợp duy nhất và lựa chọn hình; `HUONG-DAN.txt` giải thích cách đọc.

Chỉ cần vẽ lại bộ cuối từ hai thư mục dữ liệu đã hoàn thành, không chạy lại GS (gọi từ thư mục src):

```python
from pathlib import Path
from report import build_report_bundle

build_report_bundle(
    Path("../output/q-variants"),
    Path("../output/non-diag-dominance"),
    Path("../output/final"),
)
```

Các hàm `prepare_experiments()` và `run_experiments_on_size()` khi gọi riêng vẫn mặc định chỉ dùng bộ đặc; truyền `neighbors=[2,4,8]` để thêm các bộ thưa. `run_pipeline(mode, ...)` vẫn hỗ trợ cấu hình nhỏ từ Python để kiểm tra, không phải argument dòng lệnh.

## Tổ chức mã nguồn

| File trong `src` | Trách nhiệm |
| --- | --- |
| `prepare.py` | Cấu hình bộ mặc định, sinh các bộ đặc/thưa theo q và số tương tác mỗi hàng, tính q thực tế, chuẩn bị A, b, x_exact, x0. `prepare_experiments()` lần lượt trả từng `ExperimentCase`. |
| `solve.py` | Bộ giải `solve_gauss_seidel()`, sai số chuẩn vô cùng, ma trận lặp và bán kính phổ. Có thể import để giải một hệ riêng mà không nạp Matplotlib. |
| `visualization.py` | Hàm `summarize()` tổng hợp riêng từng bộ và `save_plots()` lưu các biểu đồ theo bộ cùng biểu đồ đối chiếu độ thưa. |
| `program.py` | Tự chạy lần lượt hai bộ và bước tổng hợp; nối phần chuẩn bị với bộ giải, ghi CSV/JSON. |
| `report.py` | Đọc hai bộ đã lưu, đối chiếu phần trùng, tạo ba hình và các bảng gọn cho báo cáo. |
| `utils.py` | Kiểu dữ liệu `Result` và các hàm kiểm tra mảng dùng chung. |
| `test_program.py` | Kiểm tra bộ giải, sinh dữ liệu và pipeline. |

Luồng gọi trong mỗi bộ: `program.py` -> `prepare.prepare_experiments()` -> `solve.solve_gauss_seidel()` -> lưu CSV -> `visualization.summarize()` / `visualization.save_plots()`. Sau hai bộ, `report.build_report_bundle()` đọc các file đã lưu để tạo bộ cuối.

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

Hai bộ chi tiết xuất vào `output/<mode>/`, với `<mode>` là `q-variants` hoặc `non-diag-dominance`; bộ báo cáo gọn ở `output/final/`. Các file kết quả cũ ở `outputs/` và `src/results.csv` được giữ nguyên.

- `results.csv`: các cột cũ cùng `family`, `neighbors`, `density`, `sparsity`. `family` là `dense`, `sparse_k2`, `sparse_k4` hoặc `sparse_k8`. Khóa của một trường hợp là `(n, seed, family, q_target)`.
- `convergence_history.csv`: n, seed, q_target, iteration, residual, error. Có thêm `family`, `neighbors`, `density`, `sparsity` để phân biệt lịch sử từng bộ. Error là chuẩn vô cùng của sai số nghiệm; vòng 0 là nghiệm ban đầu.
- `summary.csv`: tổng hợp riêng theo `(n, q_target, family)`, không gộp bộ đặc và các mức k. Ghi số trường hợp của từng trạng thái, tỷ lệ đạt ngưỡng dừng, trung vị và tứ phân vị số vòng lặp của các trường hợp đạt ngưỡng, thống kê bán kính phổ.
- `config.json`: cấu hình sinh dữ liệu, chuẩn và tiêu chuẩn dừng, phiên bản Python và các thư viện.
- `q_vs_iterations.png`: trung vị và khoảng tứ phân vị số vòng lặp. Trục Y của biểu đồ trên dùng thang logarit ở vùng lớn hơn 1 để các đường có số vòng lặp nhỏ không bị nén bởi giá trị lớn; vùng 0–1 vẫn tuyến tính (symlog) để biểu diễn được 0 vòng lặp. Biểu đồ tỷ lệ đạt ngưỡng phía dưới giữ nguyên thang tuyến tính. Gọi riêng `save_iteration_plot(summary, output_dir)` để vẽ lại ảnh này từ bảng thống kê.
- `q_vs_spectral_radius.png`: trung vị và khoảng tứ phân vị bán kính phổ, kèm đường rho = 1.
- `error_vs_iterations.png`: đồ thị sai số trên trục tung logarit; một seed cố định, tối đa 5 mức q cho mỗi kích thước. Giới hạn hiển thị mặc định từ `1e-12` đến `10`. Khi sai số vượt 10, đường vẽ dừng tại dấu tam giác kèm chú thích; không kéo dài thành đường ngang giả. Giá trị dưới `1e-12` chỉ được kẹp khi vẽ. CSV giữ nguyên toàn bộ dữ liệu. Có thể gọi riêng `save_error_plot(history_path, output_dir, example_seed, error_ceiling=10.0, error_floor=1e-12)` trong `visualization.py` để vẽ lại từ lịch sử có sẵn hoặc đổi giới hạn mà không chạy lại bộ giải.

Ba trạng thái: `converged` (đạt ngưỡng phần dư), `max_iterations` (hết giới hạn vòng lặp), `numerical_error` (xuất hiện số không hữu hạn). Không dùng riêng việc hết vòng lặp để kết luận phân kỳ. Giá trị NaN trong dữ liệu lỗi số học không phải sai số bằng 0.

`q < 1` là điều kiện đủ. `q >= 1` không tự động có nghĩa phân kỳ. Điều kiện hội tụ với mọi nghiệm ban đầu là `rho(B) < 1`; các giá trị rất gần 1 có thể chịu ảnh hưởng của làm tròn số thực. Không mặc định quan hệ đơn điệu giữa q và số vòng lặp cho mọi ma trận.

Các ảnh ở ngay `output/<mode>/` biểu diễn **bộ đặc**. Mỗi bộ thưa có thư mục riêng `output/<mode>/sparse_k2/`, `output/<mode>/sparse_k4/`, `output/<mode>/sparse_k8/` với cùng ba tên ảnh. Các thư mục này được tạo nếu cấu hình có trường hợp tương ứng.

Ảnh `output/<mode>/sparsity_comparison_n<N>.png` đối chiếu các bộ tại cùng kích thước N: số vòng lặp của các lần đạt ngưỡng (symlog), bán kính phổ và tỷ lệ đạt ngưỡng. Mỗi đường là một bộ; chú giải ghi k và mật độ thực tế. Không tạo ảnh đối chiếu nếu kích thước đó chỉ có bộ đặc. CSV đặc cũ vẫn có thể đọc và vẽ bằng các hàm trong `visualization.py`.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src -p "test_*.py" -v
```

Kiểm tra bao gồm cập nhật GS sau một vòng, đầu vào nguyên, chuẩn vô cùng, nghiệm ban đầu đạt ngưỡng, rho = q² cho ma trận 2 × 2 chuẩn, trường hợp q > 1 vẫn hội tụ, lỗi số học, tính tái lập và xuất dữ liệu/biểu đồ.

Các kiểm tra bổ sung xác nhận đúng k phần tử khác 0 mỗi hàng, vị trí lồng nhau, q giữ đúng, tái lập khi đổi thứ tự k, bộ đặc không đổi, thống kê không trộn bộ và xuất đủ biểu đồ đối chiếu.

Kiểm tra xác nhận không cần argument, chạy hai bộ đúng thứ tự rồi mới tổng hợp, chọn đủ q, bao gồm biên q=1, loại q<1 đúng, lưu thư mục riêng và cho kết quả giống nhau trên phần giao. Phần báo cáo được kiểm tra không đếm trùng và từ chối dữ liệu thiếu, trùng khóa hoặc không khớp cấu hình/kết quả.
