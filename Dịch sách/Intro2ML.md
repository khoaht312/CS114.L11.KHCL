# **Chương 1 – Giới thiệu**

Máy học còn được gọi là phân tích dự đoán hoặc học thống kê, nghĩa là trích xuất kiến ​​thức từ dữ liệu. Nó là một lĩnh vực nghiên cứu kết hợp của thống kê, trí tuệ nhân tạo và khoa học máy tính.

**Dùng Máy học để làm gì?**

Các vấn đề mà Máy học có thể giải quyết:

Các thuật toán học giám sát (Supervised learning algorithms): là thuật toán dự đoán đầu ra của một dữ liệu mới dựa trên các cặp dữ liệu đầu vào – đầu ra hợp lý mà người dùng cho trước. Các thuật toán học giám sát thường dễ hiểu và dễ đánh giá hiệu suất (performance). Ví dụ: Nhận dạng mã zip từ các chữ số viết tay trên phong bì, xác định xem khối u có lành tính hay không dựa trên ảnh y học, phát hiện gian lận trong giao dịch thẻ tín dụng.

Các thuật toán học không giám sát: người dùng chỉ đưa vào dữ liệu đầu vào mà không có dữ liệu đầu ra. Mặc dù các phương pháp này có nhiều ứng dụng thành công, nhưng chúng thường khó hiểu và khó đánh giá hơn. Ví dụ: Xác định chủ đề trong một tập các bài đăng trên blog, phân loại khách hàng thành các nhóm có sở thích giống nhau, phát hiện các kiểu truy cập bất thường vào một trang web.

Đối với cả việc học có giám sát và không giám sát, điều quan trọng là phải trình bày dữ liệu đầu vào như thế nào để máy tính có thể hiểu được, thông thường là dạng bảng (table). Mỗi thực thể hoặc hàng được gọi là một mẫu (sample) (hoặc điểm dữ liệu), và các cột được gọi là các đặc tính (feature).

**Hiểu được bài toán và dữ liệu của mình**

Rất có thể phần quan trọng nhất trong Máy học là hiểu và dùng dữ liệu như thế nào để giải quyết bài toán. Hãy ghi nhớ những câu hỏi sau:

• Đang trả lời những câu hỏi nào? Dữ liệu thu thập được có giải quyết được không?

• Cách biểu diễn các câu hỏi thành các vấn đề máy học?

• Dữ liệu đã thu thập đã đủ để biểu diễn bài toán chưa?

• Tôi đã trích xuất những đặc tính nào của dữ liệu, và những đặc tính này có giúp ích không?

• Làm thế nào để đánh giá mức độ thành công của ứng dụng máy học này?

• Giải pháp máy học này sẽ tương tác như thế nào với các phần còn lại trong một ứng dụng lớn?

**Dùng Python để làm gì?**

Python có các thư viện để tải dữ liệu, trực quan hóa, thống kê, xử lý ngôn ngữ tự nhiên, xử lý ảnh và hơn thế nữa. Máy học và phân tích dữ liệu về cơ bản là các quy trình lặp lại, trong đó dữ liệu thúc đẩy quá trình phân tích. Việc các quy trình này có thêm công cụ cho phép lặp lại nhanh chóng và tương tác dễ dàng là cần thiết.

**Scikit-learn**

scikit-learn là một dự án mã nguồn mở, có nghĩa là nó được sử dụng và phân phối miễn phí và bất kỳ ai cũng có thể dễ dàng lấy mã nguồn để tìm tòi học hỏi. Scikit-learn liên tục được phát triển và cải tiến, và nó có một cộng đồng người dùng rất năng động. Nó chứa rất nhiều các thuật toán học máy hiện đại nhất, cũng như các tài liệu chi tiết về từng thuật toán.

**Essential Libraries and Tools**

Jupyter Notebook, NumPy, SciPy, matplotlib, pandas, mglearn

**Python 2 và Python 3**

Python 2 hiện đã bị ngừng phát triển. Vì Python 3 có nhiều thay đổi lớn nên code Python 2 thường sẽ không chạy được trên Python 3.
 Python 3 được khuyên dùng hơn cho cả những người mới học lập trình nói riêng và tất cả lập trình viên nói chung vì đôi khi cùng một bộ mã nhưng đầu ra của Python 2 và Python 3 sẽ có đôi chút khác biệt.

**Các phiên bản được sử dụng trong quyển sách này**

Code Github của quyển sách này được thực hiện trên Python và các thư viện đã được đề cập trước đó ở các phiên bản:
 Python version: 3.5.2 |Anaconda 4.1.1 (64-bit)| (default, Jul 2 2016, 17:53:06)
 [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
 pandas version: 0.18.1
 matplotlib version: 1.5.1
 NumPy version: 1.11.1
 SciPy version: 0.17.1
 IPython version: 5.1.0
 scikit-learn version: 0.18

# **Chương 2 – Học Giám sát (Supervised Learning)**

Phương pháp học giám sát được dùng bất cứ khi nào người dùng muốn dự đoán một kết quả cụ thể từ đầu vào cho trước và ta đã có ví dụ về các cặp đầu vào/ra như mong muốn. Người dùng xây dựng 1 mô hình máy học từ những cặp đầu vào/ra này gồm cả tập huấn luyện (training set). Mục tiêu chính là đưa ra dự đoán chính xác cho các dữ liệu hoàn toàn mới. Học giám sát thường yêu cầu người dùng thủ công xây dựng tập huấn luyện, nhưng sau đó mọi thứ đều sẽ được tự động thực hiện.

**Bài toán Phân loại (Classification) và Hồi quy (Regression)**

Trong bài toán Phân loại, mục tiêu là phân loại đối tượng đầu ra thuộc lớp đối tượng nào trong danh sách các lớp đã được người dùng định nghĩa. Quá trình phân loại cũng chính là quá trình gắn nhãn cho dữ liệu đó.
 Bài toán phân loại nhị phân (binary classification) sẽ gắn nhãn đối tượng vào 1 trong 2 lớp, đây là trường hợp đặc biệt của bài toán phân loại đa lớp (multiclass classification) sẽ gắn nhãn đối tượng vào một hoặc nhiều lớp có tổng số lượng các lớp lớn hơn 2. Có thể hiểu bài toán phân loại nhị phân tương tự như việc trả lời câu hỏi có hoặc không (yes/no questions)

Trong bài toán Hồi quy, mục tiêu là dự đoán một con số liên tục, hay còn gọi là số chấm động (floating-point number) trong lập trình hoặc số thực (real number) trong toán học.

Một cách đơn giản để phân biệt bài toán Phân loại và Hồi quy là đặt câu hỏi liệu có bất kì tính liên tục (continuity) nào giữa các đối tượng đầu ra hay không. Nếu có, đây là bài toán Hồi quy.