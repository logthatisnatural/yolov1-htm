## tóm tắt bài báo 
---
### abstract
- YOLO là một pp cho bài toán object detection
- các pp trước coi đây là bài toán classification
- YOLO coi đây là bài toán regression
  - vị trí của bounding box
  - xác suất của các lớp
  - **một mạng neuron duy nhất predict bounding boxes và xác suất của nhãn cho 1 bức ảnh chỉ trong 1 lần đánh giá**
- xử lý nhanh: 45 fps
- Fast YOLO: version bé hơn đạt tốc độ 155 fps

---
### introduction 
- hệ thống YOLO gồm 3 bước
  1. resize ảnh đầu vào thành 448x488
  2. forward qua một mạng CNN
  3. giữ lại những bounding box có xác suất cao nhất (kết quả cuối cùng) 
- YOLO được train sử dụng ảnh đầy đủ
- YOLO đạt gấp đôi mAP so với các hệ thống real-time khác
- YOLO sử dụng được thông tin về toàn bộ ảnh, khác với sliding window & region proposal
- hạn chế: khó xác định được các đối tượng nhỏ

---
### Unified Detection 
- sử dụng feature từ ảnh nguyên vẹn và predict từng cái bounding box & label cho mỗi bounding box $\rightarrow$ cho phép end-to-end training  
- chia ảnh thành các ô $\text{S} \times \text{S}$
- nếu tâm của đối tượng thuộc vào ô nào, ô đó có nhiệm vụ detect đối tượng đó
- mỗi ô dự đoán $\text{B}$ bounding box và một confidence score cho mỗi cái box (model tự tin là trong cái box có chứa 1 đối tượng)
$\text{Pr(Object)} \times \text{IOU}_{\text{truth pred}}$
  - nếu ko có object nào, score = 0
  - nếu có thì ta muốn score này = IOU
- output của mỗi ô bao gồm = 5 số (cho mỗi bounding box) + $C$ class
  - bounding box 
    - $x, y$: tọa độ tâm của bounding box gióng với ô chứa nó 
    - $w, h$: chiều cao, chiều rộng của bounding box với ô chứa nó
    - confidence: IoU
  - mỗi ô
    - dự đoán đoán $C$ xác suất điều kiện $Pr(\text{Class}_i\text{|Object)}$
    - điều kiện trên ô chứa object
    - chỉ dự đoán một set xác suất các label, bất kể có bao nhiêu bounding box
  - at test time, nhân xác suất các class với từng confidence score của các box lại, được confidence score dự đoán label cho mỗi hộp $\rightarrow$ xác suất của lớp và độ chính xác của bounding box
    $\Pr(\text{Class}_i | \text{Object}) \cdot \Pr(\text{Object}) \cdot \text{IOU}_{\text{pred}}^{\text{truth}} = \Pr(\text{Class}_i) \cdot \text{IOU}_{\text{pred}}^{\text{truth}} \tag{1}$
    - prediction được encode dưới dạng một tensor với kích cỡ $\text{S} \times \text{S} \times(\text{B} \times 5 + \text{C})$
    - đánh giá trên PASCAL VOC, sử dụng $S = 7$, $B = 2$, $C = 20$, prediction là một tensor $7 \times 7 \times 30$
#### thiết kế của mạng
- đánh giá trên bộ PASCAL VOC detection dataset
- lớp tích chập để trích xuất feature
- lớp FCC dự đoán xác suất và tọa độ
- 24 tầng CNN
- 2 tầng FCC
- sử dụng tầng $1 \times 1$ reduction trước tầng tích chập $3 \times 3$
- FAST YOLO có 9 layer thay vì 24
- pre-train các lớp tích chập trên task phân loại hình ảnh ImageNet với một nửa resolution ($224 \times 224$), sau đó thì tăng gấp đôi để detect ($224 \times 224$)
- output cuối cùng là tensor với kích thước $7 \times 7 \times 30$

#### training
- pre-train 20 lớp tích chập đầu tiên trên bộ ImageNet 1000-class competition
- train 1 tuần và được accuracy 88% trên bộ đánh giá ImageNet 2012
- sử dụng framework Darknet để training và infer
- thêm 4 lớp CNN và 2 lớp FCC với trọng số khởi tạo ngẫu nhân
- tầng cuối dự đoán cả xác suất và tọa độ bounding box
  - sử dụng hàm kích hoạt tuyến tính
- tất cả các tầng còn lại sử dụng leaky ReLU 
- chuẩn hóa w và h của bounding box trong khoảng [0,1]
- loss là sum-squared eror, ko hoàn toàn align với maximize mAP
  - nó đánh trọng số lỗi xác định vị trí và lỗi phân loại, có thể không tối ưu
- vấn đề: các ô không có object sẽ có confidence score = 0, dẫn tới đạo hàm ít chú ý hơn tới những ô có object
- giải pháp: tăng loss từ dự đoán tọa độ của bounding box, và giảm loss từ dự đoán confidence cho những box không chứa đối tượng nào
  - sử dụng $\lamda_{coord} = 5$ và $\lamda_{noobj} = 0.5$
- vấn đề: metric cần reflect được là lỗi từ box nhỏ ảnh hưởng ít hơn lỗi từ box to
- giải pháp: sử dụng bình phương của w và h của bounding box thay vì w và h trực tiếp
- vấn đề: YOLO predict nhiều box, lúc training thì muốn 1 box predict 1 object
- giải pháp: chọn prediction (box) với IOU cao nhất so với ground truth
- hàm loss
  1. loss x, y
  2. loss w, h
  3. loss confidence score khi có đối tượng
  4. loss confidence score khi ko có đối tượng
  5. sai số phân loại
  - chỉ phạt lỗi phân loại nếu một đối tượng xuất hiện trong ô đó
  - phạt lỗi vị trí bounding box nếu cái predictor đó chịu trách nhiệm cho box ground truth (có IOU cao nhất)
- train 135 epoch trên bộ dữ liệu (training + validation)
- test: VOC 2007
- batch size = 64
- momentum = 0.9
- decay = 0.0005
- lr schedule
  - những epoch đầu, tăng từ $10^{-3}$ lên $10^{-2}$
  - train tiếp với $10^{-3}$ cho 75 epoch
  - $10^{-3}$ cho 30 epoch tiếp
  - $10^{-4}$ cho 30 epoch cuối
- để tránh overfit
  - dropout = 0.5 sau lớp FCC đầu tiên 
  - data augmentation: random scaling và translation tới 20% của ảnh gốc
  - chỉnh độ exposure và saturation của ảnh với factor là 1.5 trong không gian màu HSV

#### inference 
- chỉ cần đánh giá một mạng
- đánh giá nhanh
- non-maximal suppression add thêm 2-3% vào mAP   

#### hạn chế
- mỗi ô chỉ predict 2 box và có 1 class => giới hạn số lượng đối tượng model có thể pre
- ko predict chính xác các đối tượng xuất hiện theo nhóm
- khó generalize với đối tượng mới hoặc với tỷ lệ, bối cảnh mới
- hàm loss coi lỗi ở box nhỏ giống như box lớn
  - lỗi nhỏ ở box lớn thì ko sao
  - lỗi nhỏ ở box nhỏ thì có ảnh hưởng lớn đến IOU
- error chủ yếu đến từ xác định vị trí sai 

--- 
## các thuật ngữ 
- [mAP](https://www.v7labs.com/blog/mean-average-precision) (mean average precision): metric đánh giá object detection model, tìm precision trung bình cho mỗi class và lấy trung bình cộng 
- [confusion matrix](https://viblo.asia/p/tim-hieu-ve-confusion-matrix-trong-machine-learning-Az45bRpo5xY) (hơi khác một chút trong bài toán object detection): phương pháp đánh giá một classifier, đánh giá được cả những phân loại sai (acc chỉ đánh giá các phân loại đúng) 
  - true positive:  model dự đoán một label và đúng 
  - true negative: model không dự đoán label nào, và đúng
  - false positive: model dự đoán một label, nhưng ko phải là ground truth 
  - false negative: model không dự đoán label nào, và sai 
- DPM: lấy một classifier cho đối tượng $x$ và sử dụng sliding window để quét qua một ảnh tìm đối tượng đó 
- sliding window
- các kỹ thuật region proposal-based
- R-CNN: 
- IoU: phần overlap của bounding box dự đoán với bounding box ground truth
  $IoU = \frac{\text{area of overlap}}{\text{area of union}}$
- precision: đo xem tìm được bao nhiêu true positive so với positive (trong những dự đoán model cho là có label thì bao nhiêu là đúng)
$\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- recall: đo xem tìm được bao nhiêu true positive so với tất cả các prediction
  $\text{recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- ImageNet classification task
- GoogLeNet
- reduction layer
- ImageNet 1000-class competition
- average pooling layer
- FCC
- sum squared error
- diverge
- leaky ReLU
- data augmentation
- random scaling
- translation
- exposure
- saturation
- factor (điều chỉnh exposure và saturation) 
- HSV color space
- coarse features
- downsampling layers 
