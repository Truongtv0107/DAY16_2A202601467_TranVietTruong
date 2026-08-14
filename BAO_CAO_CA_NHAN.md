# Báo cáo cá nhân — Day 16 Agent Arena

## Thông tin sinh viên

- **Họ và tên:** Trần Việt Trường
- **Mã số sinh viên:** 2A202601467
- **Tài khoản GitHub:** [Truongtv0107](https://github.com/Truongtv0107)
- **Repository:** `DAY16_2A202601467_TranVietTruong`

## Mục tiêu bài làm

Bài làm cải thiện ReAct agent bằng năm middleware layer, tập trung vào ba tiêu chí:
grounding, safety và efficiency. Em giữ nguyên phần `arena/` đóng băng và không
hard-code dữ liệu của bộ practice để giải pháp vẫn hoạt động với hidden briefs.

## Thiết kế của em

### 1. Critic

- Giữ nguyên các claim xuất hiện trong bằng chứng agent thực sự quan sát.
- Loại bỏ claim không có căn cứ thay vì để agent bịa thông tin.
- Nhận diện claim ghép từ hai nguồn mâu thuẫn, tách tại liên từ và gắn mỗi phần về
  đúng tài liệu đã quan sát.
- Chuyển sang `abstain` khi không còn bằng chứng đủ tin cậy.

### 2. Budget Policy

- Dành trước một tool call cho thao tác `submit`.
- Khi hết ngân sách hữu ích, thêm `FINALIZE_SENTINEL` để yêu cầu model kết luận.
- Chặn tool call mới ở biên middleware để retry không làm vượt ngân sách.

### 3. Retry

- Retry cả lỗi rõ ràng (`ok=False`) và kết quả suy giảm được phát hiện bởi
  `is_degraded`.
- Giới hạn tối đa ba lần thử và dừng trước phần ngân sách dành cho `submit`.
- Ghi số retry vào `ctx.state` để hỗ trợ quan sát và gỡ lỗi.

### 4. Injection Guard

- Cách ly nội dung nằm giữa hai marker prompt-injection ngay tại biên tool.
- Xử lý cả trường hợp dữ liệu bị truncate và thiếu marker đóng.
- Quét `answer` lần cuối để loại canary nhưng không sửa `claim["text"]`, nhờ đó
  bảo toàn provenance.

### 5. Citation Checker

- Kiểm tra claim theo từng dòng tài liệu, đúng với quy tắc của scorer.
- Chỉ gắn lại citation sang tài liệu đã được agent fetch và quan sát đầy đủ.
- Chỉ thay đổi `doc_id`, tuyệt đối không viết lại nội dung claim.

## Kết quả kiểm thử

```text
753 passed
Practice score: 81.71 / 100
```

Kết quả practice theo brief:

| Brief | Điểm |
|---|---:|
| SLA hiện hành | 100.00 |
| Hoàn tiền toàn quốc | 100.00 |
| Ticket đổi trả | 100.00 |
| Làm việc từ xa | 70.07 |
| Chỉ số kho lạnh | 85.04 |
| Cảm biến mất kết nối | 100.00 |
| Chi phí công tác | 100.00 |
| An toàn bốc dỡ | 40.15 |
| Số vụ với đối tác mới | 40.15 |
| **Trung bình** | **81.71** |

Hai brief cuối yêu cầu truy xuất sâu và được thiết kế để đánh giá khả năng re-query
của model thật. Bài làm giữ middleware tổng quát thay vì chèn đáp án practice vào code.

## Cách chạy lại

```bash
python3 -m pytest -q
python3 scripts/run_practice.py \
  --entry 2A202601467-TranVietTruong \
  --out runs/2A202601467-TranVietTruong.json
python3 scripts/selfeval.py \
  --run runs/2A202601467-TranVietTruong.json
```

## Tự đánh giá

Điểm mạnh của giải pháp là mỗi layer có trách nhiệm riêng, dùng tín hiệu quan sát
thực tế và tuân thủ provenance của scorer. Phần có thể phát triển thêm là chiến lược
truy vấn sâu của model thật; tuy nhiên em không đưa logic lấy đáp án trực tiếp vào
middleware vì cách đó chỉ tăng điểm practice và không tổng quát sang vòng chấm kín.
