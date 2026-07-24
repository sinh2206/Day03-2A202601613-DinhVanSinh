# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng giữ chỗ dưới mỗi câu bằng câu trả lời
thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Trong hàm `call_openai`, giá trị `temperature` được chuyển thẳng tới API nên
> mức 0.0 ưu tiên cách trả lời nhất quán, còn 0.5 bắt đầu tạo ra một số thay đổi
> về câu chữ và sự kiện được lựa chọn. Ở 1.0–1.5, phản hồi thường phong phú hơn
> nhưng khả năng chọn chi tiết ít liên quan hoặc thiếu chắc chắn cũng tăng; tham
> số này điều khiển cách lấy mẫu chứ không làm kiến thức của model chính xác hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ cấu hình `temperature=0.1` cho chatbot chăm sóc khách hàng. Mức thấp
> giúp cùng một chính sách tạo ra các câu trả lời gần nhau, thuận lợi cho kiểm
> tra chất lượng và giảm nguy cơ model tự thêm điều kiện không tồn tại; sự thân
> thiện nên được quy định bằng system prompt thay vì tăng độ ngẫu nhiên.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng đầu ra mỗi ngày là `10.000 × 3 × 350 = 10,5 triệu token`. Áp dụng đúng
> `PRICING_PER_1K_TOKENS` trong bài làm, chi phí output là `$105` với GPT-4o và
> `$6,30` với GPT-4o-mini, tức GPT-4o cao hơn khoảng `16,7 lần` và chênh
> `$98,70/ngày`. Tôi dành model lớn cho phân tích khiếu nại phức tạp cần hiểu
> nhiều ngữ cảnh; các câu hỏi trạng thái đơn hàng hoặc FAQ nên chuyển cho mini.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với vai giáo viên, blockchain có thể được mô tả như một chuỗi hộp LEGO chứa
> thông tin, trong đó nhiều bạn cùng giữ bản sao nên rất khó lén sửa một hộp.
> Với vai chuyên gia tài chính, phản hồi có xu hướng dài hơn và đề cập đến hash,
> distributed ledger, consensus, finality cùng rủi ro vận hành. Trong
> `chat_with_system_prompt`, message `system` được đặt trước message `user`, vì
> vậy nó định hình mức chuyên môn và cách trình bày mà không thay đổi câu hỏi gốc.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tôi dùng đoạn 112 từ: “Trí tuệ nhân tạo đang thay đổi cách sinh viên học tập
> và giải quyết vấn đề. Thay vì chỉ tìm kiếm câu trả lời, người học có thể dùng
> mô hình ngôn ngữ để đặt câu hỏi, kiểm tra giả thuyết, nhận phản hồi và cải
> thiện sản phẩm theo từng vòng lặp. Tuy nhiên, kết quả do mô hình tạo ra vẫn
> cần được đối chiếu với nguồn đáng tin cậy. Một quy trình tốt phải kết hợp tư
> duy phản biện, kiểm thử tự động, bảo vệ dữ liệu cá nhân và ghi lại những giả
> định quan trọng trước khi đưa ứng dụng vào sử dụng thực tế.” Với model NIM
> hiện tại, `encoding_for_model` không nhận diện tên Llama nên `count_tokens`
> dùng fallback và cho `496 // 4 = 124` token; cách đếm từ cho
> `112 / 0,75 = 149,33` token, lệch khoảng `16,96%`. Đây chỉ là ước lượng:
> dấu tiếng Việt và cách tách âm tiết có thể làm một từ thành nhiều sub-token,
> nên muốn đối chiếu chi phí thật phải dùng tokenizer đúng của model phục vụ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Trong `streaming_chatbot`, mỗi `delta.content` được in ngay với `flush=True`,
> nên streaming hữu ích cho trợ lý tương tác hoặc câu trả lời dài vì giảm thời
> gian chờ cảm nhận dù tổng thời gian sinh không đổi. Non-streaming hợp lý hơn
> cho tác vụ batch, phản hồi rất ngắn, hoặc đầu ra có cấu trúc cần được parse,
> kiểm tra schema và kiểm duyệt toàn bộ trước khi ứng dụng cho người dùng thấy.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Hàm `retry_with_backoff` của tôi chờ theo công thức
> `base_delay × 2^attempt`, nên với mặc định sẽ nghỉ 0,1 rồi 0,2 rồi 0,4 giây
> thay vì liên tục gây thêm tải cho API. Delay cố định khiến nhiều client lỗi
> cùng thời điểm cũng thức dậy cùng thời điểm, tạo các đỉnh request lặp lại và
> kéo dài sự cố. Phiên bản production nên cộng thêm jitter ngẫu nhiên để phá
> tính đồng bộ giữa các client.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt tôi chọn là: “Bạn là trợ giảng thực hành AI cho sinh viên Việt
> Nam. Trả lời tối đa năm câu, ưu tiên ví dụ có thể chạy thử, và phân biệt rõ dữ
> kiện với giả định.” Giới hạn năm câu phù hợp giao diện CLI đồng thời kiểm soát
> output token; yêu cầu phân biệt giả định giúp người học biết phần nào cần tự
> kiểm chứng. Chỉ định đối tượng Việt Nam khiến thuật ngữ và ví dụ gần với lớp học.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế đáng ngại nhất là vòng lặp đang in từng chunk ra màn hình trước khi
> kiểm tra an toàn, nên nội dung không phù hợp có thể đã xuất hiện dù hệ thống
> phát hiện sau đó. Tôi sẽ kiểm tra prompt đầu vào trước khi gọi model; với chủ
> đề nhạy cảm, phản hồi được gom vào buffer, chạy bộ lọc chính sách rồi mới in.
> Các lượt thông thường vẫn có thể stream để giữ trải nghiệm nhanh, còn lịch sử
> sáu message tiếp tục giới hạn chi phí như thiết kế hiện tại.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
