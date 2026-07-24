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
> Khi `temperature` tăng từ 0.0 lên 1.5, câu trả lời thường chuyển từ ổn định,
> dễ lặp lại sang đa dạng và bất ngờ hơn về cách diễn đạt hoặc sự thật được
> chọn. Mức 1.5 có thể sáng tạo hơn nhưng cũng dễ lan man hoặc kém chính xác;
> đây là quy luật kỳ vọng của phép lấy mẫu, không phải cam kết rằng mọi lần gọi
> sẽ cho kết quả khác nhau.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi chọn khoảng `0.2`: đủ để câu trả lời tự nhiên nhưng vẫn ưu tiên tính nhất
> quán, chính xác và bám sát chính sách hỗ trợ. Với các luồng nhạy cảm như hoàn
> tiền hoặc pháp lý, tôi có thể hạ về `0.0` và kết hợp câu trả lời mẫu.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload sinh `10.000 × 3 × 350 = 10.500.000` output token/ngày. Theo bảng
> giá của đề, GPT-4o tốn khoảng `10.500 × $0,010 = $105/ngày`, còn GPT-4o-mini
> tốn `10.500 × $0,0006 = $6,30/ngày`; vì vậy GPT-4o đắt hơn khoảng `16,67`
> lần nếu chỉ xét output. GPT-4o đáng dùng cho yêu cầu phức tạp, nhiều sắc thái
> hoặc rủi ro cao; mini phù hợp hơn cho FAQ, phân loại và hội thoại số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Persona giáo viên tiểu học thường tạo câu trả lời ngắn, dùng từ phổ thông và
> ví blockchain như một cuốn sổ chung mà nhiều người cùng giữ. Persona chuyên
> gia tài chính thường trả lời dài và dùng các khái niệm như sổ cái phân tán,
> hàm băm, cơ chế đồng thuận, tính bất biến và thanh toán. System prompt vì thế
> định hướng đối tượng độc giả, độ sâu, từ vựng và kiểu ví dụ, nhưng không tự
> bảo đảm mọi dữ kiện trong phản hồi đều đúng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tôi chọn đoạn đúng 100 từ: “Việt Nam có nhiều cảnh quan đa dạng, từ núi cao
> ở phía Bắc đến những đồng bằng rộng lớn và bờ biển dài. Mỗi vùng mang một nét
> văn hóa riêng, thể hiện qua ẩm thực, trang phục, lễ hội và cách giao tiếp hằng
> ngày. Người trẻ vừa giữ gìn truyền thống, vừa sử dụng công nghệ kết nối với
> thế giới. Sự phát triển tạo ra nhiều cơ hội, nhưng cũng đặt ra yêu cầu bảo vệ
> môi trường, nâng cao chất lượng giáo dục và thu hẹp khoảng cách giữa các khu
> vực trong cả nước.” Cách đếm thô cho
> `100 / 0,75 ≈ 133,33` token; do yêu cầu không chạy chương trình, số đo
> `T = count_tokens(đoạn_văn)` cần được ghi lại khi chạy, rồi tính độ chênh là
> `|T - 133,33| / 133,33 × 100%`. Tiếng Việt có dấu và nhiều âm tiết được viết
> thành các từ tách rời; bộ mã hóa có thể chia một từ có dấu thành nhiều
> sub-token, nên số token thường cao hơn văn bản tiếng Anh có cùng số từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi model tạo câu trả lời dài hoặc có độ trễ đáng
> kể, vì người dùng thấy nội dung đầu tiên sớm và biết hệ thống vẫn hoạt động.
> Non-streaming phù hợp hơn với phản hồi ngắn, xử lý nền/batch, hoặc khi ứng
> dụng phải nhận trọn kết quả để kiểm duyệt, xác thực JSON và cập nhật giao diện
> một cách nguyên tử trước khi hiển thị.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần khoảng nghỉ, nên giảm áp lực lên dịch vụ khi lỗi
> kéo dài nhưng vẫn thử lại khá sớm nếu lỗi chỉ thoáng qua. Nếu hàng nghìn
> client cùng chờ một khoảng cố định, chúng có thể gọi lại đồng thời và tạo
> “thundering herd”, khiến máy chủ tiếp tục quá tải theo từng đợt. Trong hệ
> thống thật nên thêm jitter ngẫu nhiên vào backoff để các lần retry giãn ra.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng
> tiếng Việt; giải thích thuật ngữ bằng ví dụ thực tế và nói rõ khi chưa chắc
> chắn.” Cụm “trả lời ngắn gọn” giúp giảm thời gian đọc và chi phí output token;
> “nói rõ khi chưa chắc chắn” hạn chế cách trình bày phỏng đoán như một sự thật.
> Việc chỉ định tiếng Việt giữ trải nghiệm nhất quán với người học trong khóa.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là chỉ giữ ba lượt gần nhất, nên trợ lý nhanh chóng quên mục
> tiêu hoặc dữ kiện cũ. Tôi sẽ thêm một bản tóm tắt hội thoại: trước khi bỏ các
> message cũ, dùng model cập nhật một `conversation_summary`, rồi gửi phần tóm
> tắt này sau system prompt ở các lượt tiếp theo. Có thể bổ sung giới hạn token
> và lưu tóm tắt theo mã phiên để kiểm soát chi phí và hỗ trợ phiên dài hơn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
