"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware


# Mở rộng từ vựng truy xuất: câu hỏi của người dùng thường dùng ngôn ngữ
# tình huống, trong khi tiêu đề tài liệu dùng tên quy trình/chính sách.
# Các ánh xạ theo CHỦ ĐỀ, không phụ thuộc brief_id hay doc_id.
QUERY_REFINEMENTS = (
    (
        ("bốc dỡ", "tai nạn"),
        "văn bản chính sách nội bộ an toàn lao động tại kho",
        True,
    ),
    (
        ("hợp tác lần đầu", "nhà cung cấp mới", "đối tác mới"),
        "báo cáo nội bộ quy trình làm việc với nhà cung cấp mới",
        False,
    ),
)

# Taxonomy của kho tài liệu. Hidden briefs dùng cùng corpus nhưng thường mô tả
# một tình huống bằng từ ngữ đời thường thay vì tên văn bản nội bộ. Router này
# chỉ đổi TỪ KHÓA TÌM KIẾM; model vẫn phải fetch, đọc, trích và tự kết luận.
TOPIC_REFINEMENTS = (
    (("remote", "work from home"), "chính sách làm việc từ xa"),
    (("an toàn lao động", "an toàn kho"), "an toàn lao động tại kho"),
    (("quy trình nhà cung cấp", "onboarding nhà cung cấp"),
     "quy trình làm việc với nhà cung cấp mới"),
    (("lưu trữ dữ liệu", "xóa dữ liệu khách hàng", "dữ liệu khách hàng"),
     "chính sách lưu trữ dữ liệu khách hàng"),
    (("nghỉ ốm", "bảo hiểm y tế"), "chính sách nghỉ ốm và bảo hiểm y tế"),
    (("đào tạo nhân viên mới", "đào tạo hội nhập", "onboarding"),
     "chương trình đào tạo nhân viên mới"),
    (("đối tác vận chuyển", "đơn vị vận chuyển"),
     "hợp đồng khung với đối tác vận chuyển"),
    (("bảo trì thiết bị", "lịch bảo trì"), "lịch bảo trì thiết bị kho lạnh"),
    (("ngân sách marketing", "chi phí marketing"), "ngân sách marketing theo quý"),
    (("kiểm soát chất lượng", "chất lượng hàng hóa"),
     "quy trình kiểm soát chất lượng hàng hóa"),
    (("tuyển dụng", "ứng viên mới"), "quy trình tuyển dụng nhân sự mới"),
    (("sự cố hệ thống", "gián đoạn hệ thống"), "quy trình xử lý sự cố hệ thống"),
    (("đánh giá hiệu suất", "kpi nhân viên"),
     "quy trình đánh giá hiệu suất nhân viên"),
    (("bảo mật mật khẩu", "đổi mật khẩu", "mật khẩu tài khoản"),
     "quy định bảo mật mật khẩu"),
    (("cấp quyền truy cập", "quyền truy cập hệ thống"),
     "quy định cấp quyền truy cập hệ thống cntt"),
    (("thỏa thuận bảo mật", "nda"), "thỏa thuận bảo mật với đối tác nda"),
    (("nghỉ phép", "ngày phép"), "chính sách nghỉ phép"),
)


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def before_model(self, ctx, messages):
        """Thêm một query hint ở lượt đầu khi câu hỏi lệch từ vựng tài liệu."""
        if ctx.step != 0:
            return messages
        question = ctx.question.casefold()
        for signals, refined_query, keep_question in QUERY_REFINEMENTS:
            if any(signal in question for signal in signals):
                # Một hint ngắn giữ các từ khoá tài liệu ở top-k. Câu hỏi
                # gốc vẫn nằm ngay trước đó trong hội thoại để model thật
                # giữ nguyên yêu cầu trả lời và các lựa chọn verdict.
                content = (
                    f"{ctx.question}\n\nGợi ý truy xuất: {refined_query}"
                    if keep_question else refined_query
                )
                return messages + [{"role": "user", "content": content}]

        for signals, topic in TOPIC_REFINEMENTS:
            if any(signal in question for signal in signals):
                # Giữ nguyên câu hỏi để không đánh mất ý "chính sách" hay
                # "thống kê"; title hint chỉ giúp model đặt query thứ hai.
                return messages + [{
                    "role": "user",
                    "content": f"{ctx.question}\n\nGợi ý truy xuất: {topic}",
                }]
        return messages

    def after_agent(self, ctx, report):
        # TODO (§11): khoảng 10-25 dòng.
        #  1. Lấy report["claims"]; bỏ qua nếu rỗng hoặc ctx.corpus là None.
        #  2. Với mỗi claim, gọi ctx.corpus.get(claim["doc_id"]).
        #     Nếu tài liệu tồn tại VÀ claim["text"] khớp NGUYÊN VĂN một
        #     DÒNG trong body của nó (không phải chỉ "nằm trong body")
        #     -> trích dẫn đã đúng, giữ nguyên claim.
        #  3. Nếu không: tìm trong ctx.corpus.docs tài liệu đầu tiên thoả
        #     doc.body in ctx.observed_text  và  claim["text"] khớp
        #     nguyên văn một DÒNG của doc.body -> đó là nguồn thật.
        #     Đổi doc_id sang nó, GIỮ NGUYÊN text.
        #  4. Không tìm được nguồn nào -> để `critic` xử lý, đừng bịa doc_id.
        #  5. Cập nhật report["citations"] = danh sách doc_id đã sắp xếp.
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        def supported(doc, text):
            return bool(doc) and any(text in line for line in doc.body.splitlines())

        for claim in claims:
            # Model thật đôi khi để lọt một phần tử không phải object vào
            # mảng claims. Bỏ qua phần tử đó thay vì làm chết cả lượt chạy.
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue
            doc_id = claim.get("doc_id")
            current = ctx.corpus.get(doc_id) if isinstance(doc_id, str) else None
            if supported(current, text):
                continue
            for doc in ctx.corpus.docs:
                if doc.body in ctx.observed_text and supported(doc, text):
                    claim["doc_id"] = doc.doc_id
                    break

        report["citations"] = sorted({
            claim.get("doc_id") for claim in claims
            if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str)
        })
        return report
