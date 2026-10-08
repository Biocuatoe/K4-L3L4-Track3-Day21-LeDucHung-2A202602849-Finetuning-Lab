# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hai điều. Thứ nhất, FAILED nhưng target lại đẹp: 0,970 so với 0,765 của prompt tối ưu, vậy mà regression tụt 0,7911 → 0,6111 chỉ sau 30 step trên 225 mẫu. Thứ hai, `attn_only` — cấu hình mà tài liệu gọi là "Lỗi #1" — có train loss thấp nhất (0,5371) và target ngang `correct` (0,97). Nếu tôi chỉ nhìn loss thì sẽ kết luận vị trí adapter không quan trọng; nếu chỉ tin tài liệu thì sẽ kết luận ngược lại; số đo của tôi nói rằng với bài toán dễ và gần trần này, tôi không phân biệt được hai cách đó.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở huấn luyện. Phần tốn công nhất là phần đánh giá và hạ tầng: các lần chạy NB4 trên T4 free dao động mạnh (thời gian 30 step từ 262,2 s đến 471,5 s giữa các run trong `runs.csv`, và `docs/MEASURED-T4-2026-08-20.md` ghi cùng cấu hình chạy 1456 s rồi 1021 s), và có một lần mọi adapter đều cho target = 0,000 vì prompt lúc huấn luyện và lúc sinh khác scaffold (tôi ghi lại trong `docs/MEASURED-T4-2026-08-20.md`). Khi hoàn thiện repo, tôi còn mất thời gian vào hai lỗi chỉ có trên Windows: `verify.py` báo "eval sets unmodified" FAIL vì CRLF làm đổi checksum của các file jsonl (nội dung commit khớp checksum), và `verify.py` văng UnicodeEncodeError với console cp1252. Tôi dự đoán sẽ tốn thời gian ở LoRA config, nhưng thực tế là ở chỗ chứng minh pipeline đúng.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi từng tin "target tăng là fine-tune thành công" và "loss thấp là cấu hình tốt". Giờ tôi thấy cả hai đều sai trong chính lab này: loss xếp `attn_only` đầu bảng nhưng target chỉ hoà, và `correct` đạt 0,97 target nhưng bị FAILED ở cổng hồi quy (−0,180). Tôi cũng từng nghĩ prompt chỉ là bước đệm; baseline (b) đạt 0,765 với latency 1000,3 ms, rẻ hơn fine-tune (1398,8 ms), nên fine-tune phải chứng minh được phần 0,205 điểm còn lại đáng cái giá nó trả.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant (Claude Code) để rà soát repo, đối chiếu từng con số trong REPORT với `results/`, chuẩn bị Git LFS cho các adapter ~130 MB, chạy `pytest` / `verify.py` và soạn bản nháp REPORT/REFLECTION. Nó phải được kiểm chứng ở chỗ: bản nháp đầu có những câu mang tính suy đoán (ví dụ giải thích vì sao prompt naive làm latency cao) mà artefact không chứng minh, và `qualitative.json` không có dự đoán từng mẫu của prompt (b) nên không thể lập bảng định tính (b) vs FT như mẫu; tôi phải bỏ cột đó thay vì bịa. Ngoài ra số liệu cũ trong `docs/MEASURED-T4-2026-08-20.md` (loss 0,0549, VRAM 12,07 GB) khác `runs.csv` hiện tại (loss 0,6259, VRAM 8,78 GB) vì đó là một lần chạy trước đã bị loại; báo cáo chỉ dùng `runs.csv`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng bộ eval và đo baseline prompt tối ưu trước, như lab này. Với khách hàng thật tôi sẽ thêm hai việc mà lab này tôi đã bỏ lỡ: (1) dựng bộ regression đủ lớn và đại diện cho những gì họ cần giữ (15 câu thì một câu sai đã là 0,067), và (2) trộn 1–5% dữ liệu replay ngay từ lần huấn luyện đầu tiên, vì kết quả FAILED của tôi cho thấy thiếu replay là nguyên nhân hợp lý nhất của −0,180. Sau đó mới huấn luyện, và đặt `max_length` theo p95 đo được (98 token → ~256), không giữ mặc định 1024.
