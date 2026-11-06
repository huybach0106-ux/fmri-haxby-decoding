# Decoding loại vật thể từ fMRI, và độ chính xác đó có đáng tin không?

Dữ liệu Haxby et al. (2001): 6 người xem ảnh 8 loại vật thể (mặt người, mèo, nhà, ghế, kéo, giày,
chai, ảnh nhiễu) trong máy fMRI, 12 run, TR 2,5 s. Repo này decode loại vật thể từ vùng ventral
temporal (VT), rồi làm các kiểm tra để biết con số đó có thật không.

**Câu hỏi:** độ chính xác decoding phản ánh thông tin thật trong não, hay bị thổi phồng do cách chia
dữ liệu và do thống kê?

## Thí nghiệm

| # | Thí nghiệm | Câu hỏi | Kết quả mong đợi |
| --- | --- | --- | --- |
| 1 | Leave-one-run-out | Độ chính xác đúng là bao nhiêu? | Cao hơn hẳn may rủi 12,5% |
| 2 | Random k-fold | Chia dữ liệu sai thì sao? | Cao hơn (1) do các volume liền nhau gần như trùng nhau: đây là rò rỉ |
| 3 | Kiểm định hoán vị | Nhãn ngẫu nhiên cho bao nhiêu? | Null quanh 12,5%, p nhỏ |
| 4 | Độ trễ gán nhãn | Nhãn đã căn theo độ trễ huyết động chưa? | Nếu đã căn, dịch thêm sẽ làm giảm độ chính xác |
| 5 | Số run train | Cần bao nhiêu dữ liệu? | Tăng dần theo số run |
| 6 | Vùng não | Thông tin nằm ở đâu? | So sánh VT với vùng chọn lọc mặt / nhà |
| 7 | Ma trận nhầm lẫn | Loại nào hay bị nhầm? | Nhầm giữa các loại giống nhau về hình dạng/ngữ nghĩa |

Bộ phân loại: ridge classifier tuyến tính (alpha = 1000) trên dữ liệu đã z-score theo run. Mọi đánh
giá chính dùng leave-one-run-out, vì train và test phải ở các run khác nhau.

## Chạy

Không cần GPU. Chạy được trên Windows, macOS, Linux.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q tests         # 3 test, ~10 giây
python run.py --synthetic         # chạy thử trên dữ liệu giả lập, không cần tải
python run.py --subjects 1 2      # thử với 2 người (tải ~600 MB lần đầu)
python run.py                     # cả 6 người
```

Lần chạy đầu, nilearn tải dữ liệu vào `data/` (khoảng 300 MB mỗi người) và cache đáp ứng đã trích
vào `cache/`. Các lần sau chạy trong vài phút.

Kết quả: `results/SUMMARY.md`, các bảng `results/*.csv`, hình `figures/fig1`–`fig5`.

## Ghi chú

- Run 8 của subject 5 không có nhãn nhiệm vụ (theo tài liệu của nilearn), nên subject này có 11 run có nhãn.
- Phân tích chính dùng nhãn đúng như dataset cung cấp (`--lag 0`, như tutorial của nilearn). Trên
  subject 1, dịch thêm 2 TR làm độ chính xác giảm rõ, nên có vẻ nhãn đã được căn theo độ trễ huyết
  động. Thí nghiệm 4 báo cáo toàn bộ đường độ chính xác theo độ trễ. Không chọn độ trễ theo kết quả
  trên chính tập test rồi báo cáo con số đó như kết quả chính.
- Kiểm định hoán vị xáo nhãn trong từng run và giữ nguyên cách chia leave-one-run-out.

## Tài liệu tham khảo

- Haxby et al. Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science 2001.
- Varoquaux et al. Assessing and tuning brain decoders: cross-validation, caveats, and guidelines. NeuroImage 2017.
- Abraham et al. Machine learning for neuroimaging with scikit-learn (nilearn). Frontiers in Neuroinformatics 2014.
