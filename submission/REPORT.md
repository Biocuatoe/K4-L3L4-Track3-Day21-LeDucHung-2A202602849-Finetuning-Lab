# Lab 21 — Evaluation Report

**Họ tên**: Lê Đức Hùng  **MSSV**: 2A202602849  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Google Colab T4 16GB (fp16)

> Mọi con số dưới đây lấy từ các file trong `results/` (đã đối chiếu). Thí nghiệm chạy trên Colab T4;
> repo này chỉ chứa artefact thật đã sao chép về, không chạy lại huấn luyện.
> Kết quả cổng hồi quy là **FAILED** và được giữ nguyên, không chỉnh ngưỡng.

---

## 1. Setup

| | |
|---|---|
| Model + lý do | `unsloth/Qwen3.5-4B` — model mặc định của tier `T4`; vừa 16 GB ở fp16 (đỉnh VRAM 8,78 GB cho LoRA 16-bit), có chế độ thinking nên kiểm tra được template `<think>` |
| Dataset + lý do | Corpus mặc định của lab: **250 ticket CSKH tiếng Việt → JSON triage** (`intent`, `urgency`, `product`, `sentiment`), sinh tất định bởi `scripts/make_seed_data.py`. Không dùng dataset riêng (không có `data/CUSTOM_DATASET.md`). Lý do chọn: nhãn có cấu trúc nên chấm tự động được, và là tác vụ hẹp đúng kiểu fine-tune có thể thắng prompt |
| Train / val | **225 / 25** (seed 42) — `data/split/train.jsonl`, `val.jsonl` |
| Bộ eval đóng băng | 50 mẫu target + 15 mẫu regression (`data/eval_target.jsonl`, `data/eval_regression.jsonl`), checksum khớp `data/checksums.json` |
| `max_length` | **1024** (mặc định tier T4). p95 đo được là **98** token *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Batch | per-device 1 × gradient accumulation 16 = effective batch 16 (< 32) |
| Steps | **30 step cho cả bốn run** (2 epoch × ⌈225/16⌉) |

**Về `max_length`:** `token_stats.json` (n=250) cho mean 93,1 · p50 93 · p95 98 · p99 100 · max 101, và đề xuất `suggested_max_length = 256`. Tôi giữ **1024** theo mặc định tier T4, tức là **không** đặt theo p95 đo được mà lớn hơn ~10× p95. Với batch 1 (không padding) giá trị này không cắt mẫu nào; nhưng đúng quy trình phải đặt ~256. Tôi ghi nhận đây là một lệch so với yêu cầu 1.3.

**Template có giữ khối `<think>` không?** **Có** — `template_check.json`: `ok=true`, `open_tag_present=true`, `body_present=true`, verdict *"reasoning preserved — safe to train on traces"*. Render mẫu: `<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>\n\n4`. Tuy nhiên dữ liệu triage của lab **không có trace suy luận**: prompt huấn luyện kết thúc bằng `<think>\n\n` rỗng, phần được tính loss bắt đầu từ `</think>`.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `mask_mode` | `assistant-only` |
| `supervised_fraction` | **0,4149** (39 / 94 token) — rất xa ngưỡng 0,95 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Phần được tính loss (`supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị che (`masked_preview`): `<|im_start|>system\nPhân loại ticket sau.<|im_end|>\n<|im_start|>user\nAlo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>\n<|im_start|>assistant\n<think>\n\n`. Loss chỉ rơi trên câu trả lời JSON và token kết thúc `<|im_end|>`; system, ticket và thẻ mở `<think>` đều bị che.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Mốc đóng băng: `baselines_frozen.json` (tier T4, `eval_limit=null`, `smoke_mode=false`, SHA prompt (b) `719e74d3b6232053`, 50 target / 15 regression).

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0,000 | 0,7911 | 0,000 | 3256,2 |
| (b) base + optimized prompt | 0,765 | 0,7911 | 1,000 | 1000,3 |
| (c) LoRA fine-tune (`correct`) | **0,970** | **0,6111** | 1,000 | 1398,8 |

**(b) có mạnh hơn (a) không?** **Có**, rất rõ: target 0,000 → 0,765, format 0,000 → 1,000 và latency giảm từ 3256,2 xuống 1000,3 ms. Tôi **không** sửa `OPTIMIZED_PROMPT`; `verify.py` xác nhận SHA prompt không đổi. Vì vậy (b) là một mốc mạnh và công bằng để so với fine-tune: phần lớn khoảng cách (a)→(c) là do prompt chứ không phải do fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

Số liệu từ `runs.csv` (huấn luyện) và `autopsy.json` (đánh giá target, n=50):

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 32.464.896 | 1e-4 | 0,6259 | **0,97** | 1,0 | 401,1 | 8,78 |
| `attn_only` | q,v (2 module) | 283 (matched, α=566) | 32.456.704 | 1e-4 | 0,5371 | **0,97** | 1,0 | 262,2 | 8,79 |
| `wrong_lr` | text-linear | 16 | 32.464.896 | 1e-5 | 1,5702 | **0,00** | 0,0 | 392,2 | 8,78 |
| `qlora` (4-bit) | text-linear | 16 | 32.464.896 | 1e-4 | 0,7058 | **0,94** | 1,0 | 471,5 | 3,86 |

Mỗi run đổi đúng một biến so với `correct`: `attn_only` đổi vị trí (rank được nâng để giữ ngân sách tham số), `wrong_lr` đổi LR 1e-4 → 1e-5, `qlora` đổi độ chính xác (16-bit → 4-bit). Cả bốn đều 30 step. Xếp hạng theo **target**: `correct` = `attn_only` (0,97) > `qlora` (0,94) > `wrong_lr` (0,00).

Latency khi đo target (`autopsy.json`): `correct` 1398,8 ms · `attn_only` 907,9 ms · `wrong_lr` 5262,4 ms · `qlora` 1798,6 ms.

**4.1 — `attn_only` vs `correct`.** `attn_only` có 32.456.704 tham số huấn luyện so với 32.464.896 của `correct` (lệch 8.192 tham số ≈ 0,025%, `verify.py` báo *FAIR contrast*). Trên tập target hai run **hoà** ở 0,97, trong khi theo train loss `attn_only` lại **thấp hơn** (0,5371 < 0,6259), nghĩa là loss xếp `attn_only` đứng đầu còn target xếp ngang hàng: hai thứ tự không trùng nhau, và đó là lý do không được xếp hạng bằng loss. Điều đo được: ở ngân sách tham số khớp, nâng rank lên 283 trên q,v bù được hoàn toàn việc chỉ gắn vào 2 loại module cho bài toán triage dễ này. Tôi **không** thấy bằng chứng rằng vị trí là đòn bẩy trên tập này — chỉ 50 mẫu và gần trần 0,97 nên không đủ độ phân giải; muốn kết luận về vị trí vs rank cần bài toán khó hơn. `attn_only` cũng huấn luyện nhanh hơn (262,2 s so với 401,1 s), nhưng free T4 dao động mạnh giữa các lần chạy nên không nên diễn giải quá mức.

**4.2 — `wrong_lr`.** Chỉ đổi LR từ 1e-4 xuống 1e-5 nhưng loss cuối là 1,5702 (so với 0,6259) và model **không học được định dạng**: target 0,00, format 0,00 — giống baseline (a), đồng thời latency 5262,4 ms (cao hơn cả (a) là 3256,2 ms). Trong 30 step, LR thang full-FT gần như không dịch chuyển LoRA khỏi khởi tạo. Nếu chỉ nhìn loss mà không biết LR, ta có thể kết luận sai rằng "LoRA không học được trên dữ liệu này" hoặc "dữ liệu nhiễu", trong khi lỗi nằm ở một siêu tham số. Ở run này loss xếp hạng đúng (tệ nhất) và khớp target.

**4.3 — `qlora`.** VRAM đỉnh giảm từ 8,78 xuống 3,86 GB (**−56%**, tiết kiệm 4,92 GB). Cái giá: thời gian huấn luyện 471,5 s so với 401,1 s (**+17,6%**), loss cuối cao hơn (0,7058 so với 0,6259), target giảm từ 0,97 xuống 0,94 (−0,03, tức 1–2 mẫu trên 50; chưa đủ để kết luận có ý nghĩa thống kê) và latency suy luận cao hơn (1798,6 so với 1398,8 ms). Số đo của tôi **ủng hộ một phần** khuyến nghị tránh QLoRA cho dòng này: chi phí chất lượng có nhưng nhỏ, còn tiết kiệm bộ nhớ thật và lớn. Khi bị giới hạn VRAM thì QLoRA vẫn dùng được cho bài toán này; khi đủ 16 GB thì LoRA 16-bit đáng chọn hơn.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0,205` · `regression Δ = −0,180` · `valid_trace_rate = 0,00`
(`verdict.json`: *"general capability regressed by 0.180 (tolerance 0.020). See deck §6.3 — add 1-5% replay data."*)

**Diễn giải.** Fine-tune làm đúng việc nó được huấn luyện: so với baseline (b), target tăng từ 0,765 lên 0,970 (+0,205), format giữ 1,000. Nhưng regression tụt từ 0,7911 xuống 0,6111 (−0,180), gấp 9 lần dung sai 0,02, nên cổng hồi quy bốn nhóm không cho qua dù target rất đẹp. Đây là catastrophic forgetting điển hình: 225 mẫu huấn luyện đều cùng một dạng (ticket → JSON 4 khóa) và loss chỉ nằm trên câu trả lời JSON, nên 30 step cập nhật LoRA (32,5 triệu tham số) kéo phân phối đầu ra về phía JSON; trên 15 câu hỏi kiến thức chung (ví dụ thủ đô Việt Nam, đổi km sang mét; chấm bằng keyword recall) model trả lời kém đi. Tôi không trộn dữ liệu phổ thông (replay) nên không có gì giữ lại năng lực cũ. Lưu ý regression chỉ được đo đầy đủ cho `correct` (NB5 §4 chỉ chấm target/format cho các run đối chứng), nên không thể nói `attn_only` hay `qlora` quên ít hơn hay nhiều hơn. Các run đối chứng chứng minh tác động của LR (`wrong_lr`), vị trí/rank (`attn_only`) và độ chính xác (`qlora`) lên target, không chứng minh gì về cơ chế quên. Hàm ý: bản fine-tune là lựa chọn tốt cho **tác vụ triage** nhưng không nên deploy làm model đa dụng; muốn deploy cần chạy lại với replay 1–5% dữ liệu phổ thông, có thể giảm LR hoặc số step, và đo lại cổng. `valid_trace_rate = 0,00` nghĩa là không đầu ra nào chứa khối suy luận hợp lệ — điều dự kiến vì dữ liệu huấn luyện chỉ có `<think>` rỗng; tôi không diễn giải thêm.

---

## 6. Định tính — bắt buộc có cả ca THUA

`qualitative.json` lưu 50 mẫu target với dự đoán fine-tune (`ft_pred`, bị cắt ở 96 ký tự) và `ft_score` (tỉ lệ trong 4 khóa đúng). File **không** lưu dự đoán từng mẫu của prompt (b), nên không có cột (b) theo từng mẫu; (b) chỉ có số tổng hợp 0,765 từ `baselines_frozen.json`. Nhãn đúng lấy từ `data/eval_target.jsonl` theo chỉ số `i`.

| # | i | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | 0 | "…đặt chuột không dây… Cho tôi trả lại." | intent `doi_tra` | `doi_tra`, urgency `cao` (1,0) | ✅ FT đúng cả 4 khóa |
| 2 | 1 | "…đặt ốp lưng điện thoại… Hoàn tiền. Sớm…" | intent `hoan_tien` | `hoan_tien`, urgency `trung_binh` (1,0) | ✅ FT đúng cả 4 khóa |
| 3 | 3 | "…bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền." | `hoan_tien` / urgency `thap` | `hoan_tien` / `trung_binh` (0,75) | ❌ **FT thua**: sai urgency |
| 4 | 5 | "…nồi chiên không dầu… Thiếu phụ kiện." | `san_pham_loi` / urgency `thap` | `san_pham_loi` / `trung_binh` (0,75) | ❌ **FT thua**: sai urgency |
| 5 | 12 | "…áo khoác gió… Bị lỗi. Khi nào tiện." | `san_pham_loi` / urgency `thap` | `san_pham_loi` / `trung_binh` (0,75) | ❌ **FT thua**: sai urgency |
| 6 | 41 | "…đèn bàn LED… Giao hàng chậm. Kh…" | `van_chuyen` / urgency `thap` | `van_chuyen` / `trung_binh` (0,75) | ❌ **FT thua**: sai urgency |

Hai ca thua còn lại là i=39 ("…nồi chiên không dầu… Hoàn tiền.", gold urgency `thap`, FT `trung_binh`) và i=46 ("…đèn bàn LED… Sai màu.", gold urgency `thap`, FT `trung_binh`). Cả 6 ca thua có **cùng một mẫu**: intent và product đúng, urgency đúng là `thap` nhưng model dự đoán `trung_binh` (điểm 0,75 = 3/4 khóa đúng). Ticket mang tín hiệu khẩn rõ ("Gấp", "Khẩn", "Ngay lập tức", "Quá hạn rồi") thì model đúng; ticket không có tín hiệu rõ hoặc chỉ có cụm nhẹ như "Khi nào tiện" thì model rơi về lớp trung gian `trung_binh`. Vì `ft_pred` bị cắt nên không kiểm được `sentiment` ở các ca thua; điểm 0,75 cùng lỗi urgency nhìn thấy được khớp với đúng một khóa sai. Tổng: 44/50 mẫu điểm 1,0 và 6/50 mẫu điểm 0,75 → trung bình 0,97, khớp `autopsy.json`.

---

## 7. Các ca fine-tune thất bại

**Thất bại 1 — Quên kiến thức chung (cổng hồi quy).** `correct` có target 0,970 nhưng regression tụt 0,7911 → 0,6111 (−0,180). Nguyên nhân: tập huấn luyện chỉ gồm một tác vụ, không có replay, toàn bộ loss nằm ở JSON triage. Khắc phục: trộn 1–5% dữ liệu hỏi–đáp phổ thông (deck §6.3), hoặc giảm LR/số step/rank, rồi đo lại.

**Thất bại 2 — `wrong_lr`: không học được gì.** LR 1e-5 (thang full-FT) cho loss 1,5702, target 0,00, format 0,00, latency 5262,4 ms. Cùng dữ liệu, cùng 30 step, chỉ sai LR mà adapter vô dụng. Khắc phục: LR khoảng 10× thang full-FT cho LoRA (1e-4).

**Thất bại 3 (mức nhẹ) — sai urgency ở 6/50 mẫu** (mục 6): model thiên về lớp `trung_binh` ở các ticket thiếu tín hiệu khẩn cấp rõ ràng.

---

## 8. NB6 — merge

`merge_check.json`: điểm target trước merge **0,97**, sau merge **0,97**, `delta = 0,0`, `tolerance = 0,01`, n = 50 → merge không làm tụt điểm. Thư mục `adapters/merged/` chứa cấu hình và tokenizer (`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) nhưng **không có file trọng số** (không có `*.safetensors`) trong bản sao này, nên tôi không khẳng định gì hơn `merge_check.json`. Không có artefact nào ghi lại kiểm tra hot-swap ≥2 adapter, vì vậy **không** tuyên bố hoàn thành phần hot-swap của B1.

---

## 9. Kết luận & điều tôi học được

**Kết luận.** Không nên deploy bản fine-tune `correct` như một model đa dụng, vì cổng hồi quy FAILED: regression −0,180 so với dung sai 0,02. Nhưng với đúng việc triage ticket thì fine-tune thắng rõ: target 0,970 so với 0,765 của baseline prompt tối ưu (+0,205) và 0,000 của prompt naive, format hợp lệ 100%. Cái giá là latency 1398,8 ms so với 1000,3 ms của (b) (khoảng +40%) và, quan trọng hơn, năng lực chung bị suy giảm. Lập luận nhân quả của tôi: dữ liệu huấn luyện đồng nhất (225 mẫu, một tác vụ) cộng với mask `assistant-only` chỉ tính loss trên JSON làm các cập nhật LoRA chuyên biệt hoá phân phối đầu ra mà không có tín hiệu nào giữ lại kiến thức khác, nên regression sụt; trộn replay 1–5% nhiều khả năng giảm đáng kể điều này, nhưng tôi chưa đo nên không khẳng định. Về đòn bẩy: learning rate là đòn bẩy rõ nhất trong lab này (`wrong_lr` từ 0,97 xuống 0,00 chỉ vì một con số); mask đúng là điều kiện cần (supervised_fraction 0,4149, không phải ≥0,95); vị trí adapter không phân biệt được giữa `attn_only` và `correct` trên tập này (cùng 0,97 khi ngân sách tham số khớp); QLoRA đổi 56% VRAM lấy 0,03 target và +17,6% thời gian. Hạn chế: tập target chỉ 50 mẫu, regression chỉ 15 mẫu (một câu sai ≈ 0,067) và mỗi cấu hình chỉ chạy một lần nên các chênh lệch nhỏ (như 0,97 vs 0,94) không đủ ý nghĩa thống kê. Quyết định của tôi: ghi nhận FAILED là kết quả hợp lệ, không nới ngưỡng, và làm lại với replay trước khi nghĩ đến deploy.

**Ba điều tôi học được**
1. Không xếp hạng bằng loss: `attn_only` có loss thấp nhất (0,5371) nhưng target chỉ ngang `correct` (0,97, loss 0,6259). Loss huấn luyện không thay được chỉ số đánh giá thật.
2. Target tăng không có nghĩa là an toàn: +0,205 target đi cùng −0,180 regression; chỉ khi đo cả nhóm regression mới thấy quên thảm hoạ.
3. Một siêu tham số sai (`wrong_lr`) đủ biến adapter thành vô dụng (target 0,00) dù mọi thứ khác đúng; phải kiểm tra kết quả sinh ra chứ không chỉ đường loss.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (a) huấn luyện lại `correct` với 1–5% dữ liệu replay và đo lại cổng; (b) hạ `max_length` về 256 theo p95; (c) chấm regression cho cả `attn_only` và `qlora`; (d) mở rộng tập target/regression để phân biệt được các chênh lệch nhỏ; (e) chạy hot-swap hai adapter và lưu bằng chứng.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap — chỉ có `merge_check.json` (delta 0,0 ≤ 0,01); **chưa** có bằng chứng hot-swap ≥2 adapter nên không tuyên bố đủ B1
- [ ] B2 dataset miền riêng — không làm
- [ ] B3 reasoning-trace collapse — không làm (chỉ một `MASK_MODE`)
- [ ] B4 quét rank có kiểm soát — không làm
- [ ] B5 HuggingFace Hub — không làm
