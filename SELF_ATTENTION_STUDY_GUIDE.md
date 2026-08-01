# Self-Attention — Hướng dẫn ôn vấn đáp theo notebook

Tài liệu này đi cùng `Self_attention_demo.ipynb`. Số cell dưới đây là vị trí
thực tế trong notebook, bắt đầu từ 0. Các comment như `Cell 1`, `Cell 3` trong
code chỉ là nhãn demo cũ, không phải chỉ số cell Jupyter.

## 1. Cơ chế cần thuộc trước khi mở notebook

Với biểu diễn đầu vào \(X \in \mathbb{R}^{B \times L \times d_{model}}\):

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

\[
S=\frac{QK^\top}{\sqrt{d_k}},\qquad
A=\operatorname{softmax}(S+\text{mask}),\qquad
H=AV
\]

Trong multi-head attention:

\[
\operatorname{MHA}(X)
=\operatorname{Concat}(H_1,\ldots,H_h)W_O
\]

Với mBERT trong notebook:

- `B = 1`: mỗi lần demo một câu.
- `L = seq_len`: số token sau WordPiece, có cả `[CLS]` và `[SEP]`.
- `d_model = 768`: kích thước hidden state.
- `h = 12`: số attention head trong một layer.
- `d_k = d_v = 768 / 12 = 64`: kích thước mỗi head.
- `num_hidden_layers = 12`: số Transformer encoder block.
- Một attention matrix của một layer có shape
  `(batch, heads, query_length, key_length)`.

### Vai trò Q, K, V

- **Query:** token hiện tại đang muốn tìm thông tin gì.
- **Key:** mỗi token cung cấp “nhãn” để query so khớp.
- **Value:** nội dung được trộn theo trọng số sau khi query so khớp với key.
- `QKᵀ` tạo điểm tương thích; `softmax` biến mỗi hàng thành phân phối có tổng
  xấp xỉ 1; nhân với `V` tạo biểu diễn mới chứa thông tin ngữ cảnh.

### Vì sao chia cho \(\sqrt{d_k}\)?

Khi `d_k` lớn, tích vô hướng có độ lớn/variance tăng. Không scale sẽ làm
softmax dễ bão hòa về gần 0 hoặc 1, gradient nhỏ và việc học kém ổn định.

### BERT khác GPT ở attention mask

- BERT encoder dùng attention hai chiều: token có thể nhìn cả bên trái lẫn bên
  phải, chỉ mask padding nếu có.
- GPT decoder dùng causal mask: token ở vị trí `t` không được nhìn token tương
  lai. Vì vậy notebook này minh họa self-attention nhưng không minh họa causal
  attention của mô hình sinh tự hồi quy.

### Giới hạn diễn giải

Attention weight là pattern nội bộ, không tự động là lời giải thích nhân quả.
Một head cho “Nó” trọng số cao tới “mèo” chưa đủ chứng minh model đã giải đúng
coreference. Cần kiểm tra nhiều câu, nhiều head/layer và tốt hơn là có phép
can thiệp/ablation.

---

## 2. Giải thích và câu hỏi dự đoán theo từng cell

## Cell 0 — Markdown: Google Colab Setup

**Mục đích:** báo rằng cell kế tiếp cài phụ thuộc. Không có tham số chạy.

**Thầy có thể hỏi**

1. **Tại sao phải setup trước?**  
   Vì `bertviz`, `transformers` và `torch` không phải thư viện chuẩn của Python.
2. **Có gọi API trả phí không?**  
   Không. Notebook tải model open-source từ Hugging Face và inference cục bộ.

## Cell 1 — Cài thư viện

```python
%pip install -q bertviz transformers torch
```

**Tham số/thành phần**

- `%pip`: Jupyter magic, cài package vào đúng Python environment của kernel.
  An toàn hơn `!pip`, vì `!pip` đôi khi trỏ sang interpreter khác.
- `install`: hành động cài package.
- `-q`: quiet, giảm log.
- `bertviz`: giao diện trực quan hóa attention.
- `transformers`: tokenizer, model và cấu hình Hugging Face.
- `torch`: tensor, inference và phép toán model.

**Thầy có thể hỏi**

1. **Vì sao dùng `%pip` thay `!pip`?**  
   `%pip` được IPython gắn với kernel hiện tại, giảm lỗi cài đúng package nhưng
   notebook vẫn không import được.
2. **Có cần chạy cell này mỗi lần không?**  
   Chỉ cần khi runtime mới/chưa có package; Colab reset runtime thì phải cài lại.
3. **Cài package có đồng nghĩa model đã tải chưa?**  
   Không. Trọng số model được tải ở `from_pretrained` trong cell 3.

## Cell 2 — Markdown: mục tiêu và chuẩn bị

**Mục đích:** giới thiệu mBERT, BertViz, cách mở môi trường và giới hạn tải
model khoảng 714 MB. Không có tham số runtime.

**Thầy có thể hỏi**

1. **Vì sao chọn multilingual BERT?**  
   Vocabulary và pretraining của nó bao phủ nhiều ngôn ngữ, trong đó có tiếng
   Việt, nên có thể demo cùng một model cho tiếng Việt và tiếng Anh.
2. **“Cased” nghĩa là gì?**  
   Tokenizer/model phân biệt chữ hoa và chữ thường.
3. **BertViz có phải model không?**  
   Không; nó chỉ hiển thị attention tensors do model trả về.

## Cell 3 — Load tokenizer và model

```python
MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True,
    attn_implementation="eager",
)
model.eval()
```

**Tham số/thành phần**

- `MODEL_NAME`: Hugging Face model ID.
- `AutoTokenizer`: tự chọn tokenizer class từ model config.
- `from_pretrained(MODEL_NAME)`: tải config, vocabulary và trọng số đã huấn luyện.
- `AutoModel`: tải phần backbone BERT, không thêm classification head.
- `output_attentions=True`: yêu cầu trả attention của mọi layer.
- `attn_implementation="eager"`: dùng triển khai PyTorch eager để attention
  weights được materialize; một số optimized kernels không trả matrix này.
- `model.eval()`: tắt hành vi training như dropout; không tắt gradient.
- `p.numel()`: số phần tử của một parameter tensor.
- `sum(...)`: tổng số tham số trong toàn model.

**Thầy có thể hỏi**

1. **`eval()` khác `no_grad()` thế nào?**  
   `eval()` đổi hành vi layer như dropout; `no_grad()` ngừng lưu computation
   graph. Demo inference nên dùng cả hai.
2. **Tại sao load report có `UNEXPECTED` keys?**  
   Checkpoint có thể chứa head pretraining như MLM/NSP, còn `AutoModel` chỉ lấy
   backbone; các key head bị bỏ là bình thường trong trường hợp này.
3. **178 triệu tham số có phải đều nằm trong attention?**  
   Không. Nó còn gồm word/position/type embeddings, feed-forward layers,
   LayerNorm và pooler; vocabulary đa ngôn ngữ lớn chiếm nhiều tham số.
4. **Tại sao 12 head nhưng hidden size vẫn 768?**  
   Mỗi head làm việc ở 64 chiều; 12 output được concat về 768 rồi chiếu qua
   `W_O`.

## Cell 4 — Markdown: output kỳ vọng và câu demo

**Mục đích:** cho mốc kiểm tra 12 layer, 12 head và khoảng 177,85 triệu tham
số; giới thiệu câu có đại từ “Nó”. Không có tham số chạy.

**Thầy có thể hỏi**

1. **Tổng số attention head có phải 144 không?**  
   Có 12 head/layer × 12 layer = 144 head instances, nhưng không phải một
   attention operation duy nhất có 144 head song song.
2. **Tại sao output tokenizer có thể khác tài liệu?**  
   Phiên bản tokenizer/vocabulary handling có thể thay đổi cách hiển thị
   subword; cần dựa vào output thực tế.

## Cell 5 — Tokenize, forward pass và lấy attention

```python
inputs = tokenizer(
    sentence,
    return_tensors="pt",
    return_offsets_mapping=True,
    truncation=True,
)
offsets = inputs.pop("offset_mapping")[0].tolist()
```

**Tham số/thành phần**

- `sentence`: chuỗi đầu vào.
- `return_tensors="pt"`: trả tensor PyTorch và thêm batch dimension.
- `return_offsets_mapping=True`: trả cặp `(char_start, char_end)` cho mỗi token,
  dùng nối subword về từ gốc.
- `truncation=True`: cắt câu nếu vượt giới hạn context của tokenizer/model.
- `pop("offset_mapping")`: lấy offsets ra vì model không nhận đối số này.
- `[0]`: lấy sample đầu tiên của batch một phần tử.
- `.tolist()`: đổi tensor thành list Python.
- `inputs["input_ids"][0]`: chuỗi token ID của sample đầu tiên.
- `convert_ids_to_tokens`: đổi ID sang token string để hiển thị.
- `torch.no_grad()`: không tạo gradient graph, giảm bộ nhớ inference.
- `model(**inputs)`: unpack dictionary thành các keyword arguments như
  `input_ids`, `token_type_ids`, `attention_mask`.
- `outputs.attentions`: tuple dài 12, mỗi phần tử ứng với một layer.

**Giải thích shape**

`(1, 12, 15, 15)` nghĩa là:

- 1 câu trong batch;
- 12 head;
- 15 query token;
- mỗi query phân phối attention lên 15 key token.

**Thầy có thể hỏi**

1. **Tại sao “mèo” thành `m` và `##èo`?**  
   WordPiece dùng subword; `##` báo token tiếp nối token trước trong cùng từ.
2. **`attention_mask` có phải causal mask không?**  
   Không trong BERT. Ở đây nó chủ yếu phân biệt token thật và padding.
3. **Mỗi hàng attention có tổng bằng bao nhiêu?**  
   Xấp xỉ 1 sau softmax, với sai số floating point.
4. **Tokenizer có tạo positional embedding không?**  
   Tokenizer tạo token IDs; model tự tạo/tra position IDs và cộng position
   embeddings. Self-attention tự nó không biết thứ tự nếu thiếu thông tin vị trí.

## Cell 6 — Markdown: output tokenization

**Mục đích:** hiển thị token và shape kỳ vọng. Không có tham số chạy.

**Thầy có thể hỏi**

1. **`[CLS]` và `[SEP]` để làm gì?**  
   `[CLS]` là token tổng hợp đầu chuỗi; `[SEP]` đánh dấu kết thúc/phân cách
   segment.
2. **Special token có tham gia attention không?**  
   Có; chúng có query/key/value như các token khác và có thể nhận trọng số lớn.

## Cell 7 — BertViz Head View

```python
head_view(attention, tokens, layer=8)
```

**Tham số**

- `attention`: tuple attention của 12 layer.
- `tokens`: nhãn token khớp với hai chiều cuối của tensor.
- `layer=8`: chọn layer index 8, tức layer thứ 9 vì Python đếm từ 0.

**Thầy có thể hỏi**

1. **Đường đậm hơn nghĩa là gì?**  
   Attention weight lớn hơn trong head/layer đang chọn, không đồng nghĩa quan
   hệ nhân quả mạnh hơn.
2. **Tại sao đổi head thì pattern đổi?**  
   Mỗi head có projection matrices `W_Q`, `W_K`, `W_V` riêng.
3. **Vì sao chọn layer 8?**  
   Đây là lựa chọn để quan sát, không phải layer “đúng” duy nhất; nên khảo sát
   nhiều layer/head trước khi kết luận.

## Cell 8 — Markdown: chuyển sang kiểm chứng số

**Mục đích:** nhắc rằng màu/đường trong visualization cần được đối chiếu bằng
giá trị tensor. Không có tham số.

**Thầy có thể hỏi**

1. **Tại sao cần xem số thay vì chỉ nhìn hình?**  
   Thị giác dễ bị scale và số lượng đường đánh lừa; số cho phép so sánh rõ.

## Cell 9 — Đo attention từ “Nó” tới span từ

**Tham số/thành phần**

- `token_span(text)`: tìm char span lần xuất hiện đầu tiên bằng
  `sentence.index(text)`.
- `offsets`: nối char span với token indices.
- `no_indices`, `meo_indices`, `ban_indices`: toàn bộ subword của từng cụm.
- `layer = 8`: index 8 = layer thứ 9.
- `attention[layer][0]`: chọn layer và batch đầu tiên, còn tensor
  `(num_heads, query, key)`.
- `attn_layer[head][no_indices]`: chọn hàng query thuộc “Nó”.
- `[:, meo_indices]`: chọn cột key thuộc toàn bộ subword “mèo”.
- `.sum(dim=1)`: cộng attention mass của các target subword.
- `.mean()`: nếu query có nhiều subword, lấy trung bình các hàng query.
- `.item()`: đổi scalar tensor thành Python float.
- `attn_layer.shape[0]`: số head thực tế, không hard-code 12.

**Thầy có thể hỏi**

1. **Chiều nào là query, chiều nào là key?**  
   Hai chiều cuối là `[query_position, key_position]`; code đọc hàng “Nó” và
   cột “mèo”/“bàn”.
2. **Tại sao phải cộng cả `m` và `##èo`?**  
   Một từ gốc có thể gồm nhiều subword; chỉ dùng token đầu sẽ bỏ một phần
   attention mass và làm so sánh phụ thuộc tokenization.
3. **Nếu “mèo” lớn hơn “bàn” ở 10/12 head thì model chắc chắn hiểu “Nó” chưa?**  
   Chưa. Đây chỉ là correlation trong raw attention của một layer.
4. **Tại sao không cộng trên tất cả layer/head?**  
   Các head/layer có vai trò và scale pattern khác nhau; cộng tùy tiện có thể
   làm mất cấu trúc. Muốn aggregate cần định nghĩa phương pháp rõ ràng.

## Cell 10 — Markdown: giới thiệu Model View

**Mục đích:** chuyển từ một layer sang toàn bộ 12 × 12 head. Không có tham số.

**Thầy có thể hỏi**

1. **Head View và Model View khác nhau thế nào?**  
   Head View tập trung vào token connections ở layer/head chọn; Model View cho
   phép duyệt toàn kiến trúc và so sánh rộng hơn.

## Cell 11 — BertViz Model View

```python
model_view(attention, tokens)
```

**Tham số**

- `attention`: attention tuple của tất cả layer.
- `tokens`: nhãn token theo đúng `seq_len`.
- Không truyền `layer`, vì Model View cần toàn bộ layers.

**Thầy có thể hỏi**

1. **Model View có hiển thị Q, K, V không?**  
   Không trực tiếp; nó hiển thị weights sau scaled dot-product và softmax.
2. **Có thể so attention giữa head chỉ bằng độ đậm không?**  
   Có thể quan sát, nhưng kết luận chức năng cần nhiều dữ liệu và phân tích số.

## Cell 12 — Markdown: so sánh ngôn ngữ

**Mục đích:** đặt giả thuyết về sự khác nhau do WordPiece vocabulary và câu
đầu vào. Không có tham số.

**Thầy có thể hỏi**

1. **Nhiều token hơn có nghĩa câu khó hơn không?**  
   Không nhất thiết; nó có thể chỉ phản ánh vocabulary có ít whole-word token
   phù hợp với ngôn ngữ đó.

## Cell 13 — So sánh tiếng Việt và tiếng Anh

**Tham số/thành phần**

- `sentences`: dictionary ánh xạ nhãn hiển thị → câu.
- `.items()`: lặp từng cặp nhãn/câu.
- `return_tensors="pt"` và `truncation=True`: như cell 5.
- `outputs_i.attentions`: attention riêng cho từng câu.
- `layer=9`: layer index 9, tức layer thứ 10.

**Thầy có thể hỏi**

1. **Tại sao tiếng Việt có 25 token còn tiếng Anh 21?**  
   mBERT WordPiece tách nhiều từ/tên riêng tiếng Việt thành subword hơn.
2. **Có thể kết luận model hiểu tiếng Anh tốt hơn chỉ từ số token không?**  
   Không. Cần benchmark downstream; token count chỉ là một yếu tố.
3. **Hai heatmap có so số tuyệt đối trực tiếp được không?**  
   Cần thận trọng vì độ dài chuỗi và token boundaries khác nhau; softmax phân
   phối mass trên số key khác nhau.

## Cell 14 — Markdown: demo tương tác

**Mục đích:** báo phần bonus cho phép thay câu và layer. Không có tham số chạy.

**Thầy có thể hỏi**

1. **Muốn tạo thí nghiệm tốt cần thay gì?**  
   Chỉ thay một yếu tố trong câu, giữ các yếu tố khác gần giống để so sánh.

## Cell 15 — Hàm `visualize_attention`

```python
def visualize_attention(sentence, layer=9):
```

**Tham số**

- `sentence`: câu cần tokenize và chạy model.
- `layer=9`: mặc định layer index 9, tức layer thứ 10.
- Điều kiện `0 <= layer < num_hidden_layers`: ngăn index âm hoặc vượt 11.
- `raise ValueError(...)`: báo lỗi đầu vào rõ ràng.
- `truncation=True`: tránh vượt max position.
- Các lệnh tokenize, `no_grad`, forward và `head_view`: giống cell 5/7.

**Các câu thử**

- Đại từ mơ hồ: quan sát pattern, không kỳ vọng một đáp án duy nhất.
- Phủ định: xem token “không” nhận/gửi attention thế nào.
- Context xa: kiểm tra liên kết qua ranh giới câu.
- Tên riêng: quan sát WordPiece phân mảnh `VinFast`, `Xanh`.

**Thầy có thể hỏi**

1. **Tại sao validate layer?**  
   Model chỉ có index 0–11; validation biến lỗi index khó hiểu thành thông báo
   có ý nghĩa.
2. **Nếu câu dài hơn giới hạn thì sao?**  
   `truncation=True` cắt phần dư; nếu thông tin quan trọng nằm cuối, kết quả demo
   có thể sai mục tiêu và cần chiến lược chunking.
3. **Muốn so ảnh hưởng của phủ định nên làm gì?**  
   So cặp tối thiểu như “Tôi thích ăn cơm” và “Tôi không thích ăn cơm”, cùng
   layer/head, thay vì chỉ xem một câu.

## Cell 16 — Markdown: takeaways, checklist, tài liệu

**Mục đích:** tổng kết có điều kiện, nhắc chuẩn bị môi trường và nguồn tham
khảo. Không có tham số runtime.

**Thầy có thể hỏi**

1. **Có được gọi mỗi head là head ngữ pháp/đại từ không?**  
   Chỉ nên gọi là giả thuyết sau khi quan sát; muốn khẳng định cần thống kê trên
   dataset và phép can thiệp.
2. **Có biết chính xác GPT-4 có bao nhiêu layer/head/tham số không?**  
   Không nên suy đoán nếu nhà cung cấp không công bố. Cơ chế họ Transformer
   không cho phép suy ra cấu hình cụ thể.
3. **BertViz chứng minh được điều gì?**  
   Nó cho thấy attention weights model tạo ra cho input cụ thể; nó hỗ trợ phân
   tích nhưng không chứng minh reasoning hay quan hệ nhân quả.

---

## 3. Mười câu thầy dễ hỏi nhất

1. **Self-attention là gì?**  
   Mỗi token tạo query, so với keys của các token, softmax thành trọng số rồi
   trộn values để có biểu diễn phụ thuộc ngữ cảnh.
2. **Q/K/V có phải ba bản sao giống nhau không?**  
   Cùng xuất phát từ hidden states nhưng qua ba ma trận học được khác nhau.
3. **Vì sao gọi là “self”?**  
   Q, K, V đến từ cùng một sequence; cross-attention lấy Q và K/V từ hai nguồn.
4. **Vì sao multi-head tốt hơn một head?**  
   Nhiều projection subspace cho phép mô hình học nhiều pattern song song.
5. **Complexity theo độ dài chuỗi?**  
   Attention matrix có thời gian/bộ nhớ bậc hai theo `L`: thường nói
   \(O(L^2d)\) cho tính toán và \(O(L^2)\) cho weights mỗi head.
6. **Attention có biết thứ tự từ không?**  
   Không nếu đứng một mình; BERT cộng positional embeddings vào input.
7. **BERT và GPT khác nhau ở đâu trong attention?**  
   BERT hai chiều; GPT causal, không nhìn token tương lai.
8. **Attention weight có phải xác suất từ nào là đáp án không?**  
   Không; nó là phân phối trộn values trong một head/layer.
9. **Tại sao phải dùng subword span?**  
   Một từ có thể bị tách nhiều token; đo một token sẽ bỏ thông tin.
10. **`eval()` và `no_grad()` có thay nhau được không?**  
    Không. Một cái đổi hành vi layer, một cái tắt autograd.

## 4. Trình tự trình bày 90 giây

1. Nói công thức `Q=XWQ`, `K=XWK`, `V=XWV`.
2. Nói `softmax(QKᵀ / sqrt(d_k))` tạo attention weights.
3. Nói weights nhân `V` để cập nhật biểu diễn token.
4. Chỉ shape `(1, 12, 15, 15)` và giải thích bốn chiều.
5. Mở Head View, chọn một head/layer và chỉ query→key.
6. Mở cell số liệu, giải thích phải cộng cả subword của “mèo”.
7. Kết luận có điều kiện: visualization cho pattern, không phải bằng chứng
   nhân quả hay bảo đảm model giải coreference.
