VIETNAMESE_VOWELS = set('aăâeêioôơuưyAĂÂEÊIOÔƠUƯY')

VIETNAMESE_VOWELS_ALL = set(
    'aăâeêioôơuưyAĂÂEÊIOÔƠUƯY'
    'àằầèềìòồờùừỳÀẰẦÈỀÌÒỒỜÙỪỲ'
    'áắấéếíóốớúứýÁẮẤÉẾÍÓỐỚÚỨÝ'
    'ảẳẩẻểỉỏổởủửỷẢẲẨẺỂỈỎỔỞỦỬỶ'
    'ãẵẫẽễĩõỗỡũữỹÃẴẪẼỄĨÕỖỠŨỮỸ'
    'ạặậẹệịọộợụựỵẠẶẬẸỆỊỌỘỢỤỰỴ'
)

BASE_VOWELS = set('aeiouyAEIOUY')

VIETNAMESE_CONSONANTS = set('bcdđghklmnpqrstvxBCDĐGHKLMNPQRSTVX')

VALID_INITIAL_CLUSTERS = {
    'ch', 'gh', 'gi', 'kh', 'ng', 'ngh', 'nh', 'ph', 'qu', 'th', 'tr',
    'Ch', 'Gh', 'Gi', 'Kh', 'Ng', 'Ngh', 'Nh', 'Ph', 'Qu', 'Th', 'Tr',
    'CH', 'GH', 'GI', 'KH', 'NG', 'NGH', 'NH', 'PH', 'QU', 'TH', 'TR',
    'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'kl', 'kr',
    'pl', 'pr', 'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw',
}

VALID_FINAL_CONSONANTS = {
    'c', 'ch', 'm', 'n', 'ng', 'nh', 'p', 't',
    'C', 'CH', 'M', 'N', 'NG', 'NH', 'P', 'T'
}

TONE_MARKS = {
    '\u0300': 'grave',
    '\u0301': 'acute',
    '\u0303': 'tilde',
    '\u0309': 'hook',
    '\u0323': 'dot_below',
}

MODIFIER_MARKS = {
    '\u0302': 'circumflex',
    '\u0306': 'breve',
    '\u031B': 'horn',
}

CIRCUMFLEX_BASES = set('aeoAEO')
BREVE_BASES = set('aA')
HORN_BASES = set('ouOU')

VIETNAMESE_CHARS_WITH_DIACRITICS = set(
    'àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵđ'
    'ÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴĐ'
)

COMMON_VIETNAMESE_WORDS = {
    'tôi', 'tao', 'mình', 'chúng', 'ta', 'bạn', 'anh', 'chị', 'em', 'ông', 'bà',
    'cô', 'chú', 'bác', 'họ', 'nó', 'hắn', 'cậu', 'mày', 'người', 'ai', 'gì',
    'là', 'có', 'được', 'làm', 'đi', 'đến', 'về', 'ra', 'vào', 'lên', 'xuống',
    'ăn', 'uống', 'ngủ', 'nói', 'nghe', 'nhìn', 'thấy', 'biết', 'hiểu', 'yêu',
    'ghét', 'muốn', 'cần', 'phải', 'nên', 'cho', 'lấy', 'mang', 'đem', 'gửi',
    'viết', 'đọc', 'học', 'dạy', 'chơi', 'hát', 'múa', 'vẽ', 'nấu', 'giặt',
    'nghĩ', 'nhớ', 'quên', 'tìm', 'kiếm', 'chờ', 'đợi', 'gặp', 'chia', 'ly',
    'nhà', 'người', 'việc', 'đời', 'năm', 'tháng', 'ngày', 'giờ', 'phút', 'giây',
    'nước', 'đất', 'trời', 'mây', 'mưa', 'nắng', 'gió', 'bão', 'sông', 'núi',
    'biển', 'rừng', 'cây', 'hoa', 'lá', 'quả', 'con', 'cái', 'chiếc', 'quyển',
    'sách', 'vở', 'bút', 'bàn', 'ghế', 'cửa', 'đường', 'phố', 'chợ', 'trường',
    'đẹp', 'xấu', 'tốt', 'lớn', 'nhỏ', 'cao', 'thấp', 'dài', 'ngắn',
    'rộng', 'hẹp', 'nặng', 'nhẹ', 'nóng', 'lạnh', 'ấm', 'mát', 'khô', 'ướt',
    'mới', 'cũ', 'trẻ', 'già', 'giàu', 'nghèo', 'khỏe', 'yếu', 'vui', 'buồn',
    'rất', 'lắm', 'quá', 'hơi', 'khá', 'cũng', 'đã', 'đang', 'sẽ', 'vẫn',
    'còn', 'mới', 'vừa', 'từng', 'hay', 'thường', 'luôn', 'chưa', 'không',
    'của', 'với', 'trong', 'ngoài', 'trên', 'dưới', 'trước', 'sau', 'bên',
    'và', 'hoặc', 'hay', 'nhưng', 'mà', 'nếu', 'thì', 'vì', 'nên', 'để',
    'nào', 'đâu', 'khi', 'sao', 'bao', 'mấy', 'bằng',
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'trăm', 'nghìn', 'ngàn', 'triệu', 'tỷ',
    'hôm', 'nay', 'qua', 'mai', 'kia', 'tuần', 'mùa', 'xuân',
    'hạ', 'thu', 'đông', 'sáng', 'trưa', 'chiều', 'tối', 'đêm',
    'quân', 'tử', 'từ', 'vị', 'kiến', 'chân', 'thực', 'trì', 'đạo', 'đức',
    'nhân', 'nghĩa', 'lễ', 'trí', 'tín', 'hiếu', 'thuận', 'tiết', 'nghĩa',
    'tâm', 'thân', 'khẩu', 'ý', 'thiện', 'ác', 'phúc', 'họa', 'mệnh', 'số',
    'chân', 'quán', 'quy', 'tài', 'tai', 'phạt', 'kỳ', 'điều', 'tuấn', 'bị',
    'nhục', 'phẫn', 'phần', 'chức', 'như', 'cơ', 'bao', 'thắng', 'đội',
    'xót', 'thầm', 'cao', 'sơn', 'lưu', 'thủy', 'thực', 'trì',
    'đau', 'đớn', 'nhẽ', 'thiếp', 'xa', 'xôi', 'thấu', 'tình', 'chẳng',
    'dương', 'chi', 'bát', 'thức', 'tân', 'bí', 'chỉ', 'nơi', 'hầm', 'sâu',
    'yểm', 'lệ', 'bất', 'dữ', 'ngã', 'thù', 'thân', 'thần', 'hoài', 'hoàn',
    'hạt', 'nguyệt', 'dư',
    'tử', 'từ', 'vị', 'kỳ', 'chi', 'giả', 'dã', 'hồ', 'tai', 'tài',
    'đại', 'tiểu', 'thượng', 'hạ', 'trung', 'ngoại', 'nội', 'tiền', 'hậu',
    'đông', 'tây', 'nam', 'bắc', 'thiên', 'địa', 'nhân', 'vật', 'sự', 'lý',
    'vì', 'sao', 'được', 'thế', 'này', 'kia', 'đó', 'đây', 'ấy', 'nọ',
    'thôi', 'rồi', 'xong', 'hết', 'còn', 'lại', 'nữa', 'thêm', 'bớt',
}

CHAR_CONFUSIONS = {
    'l': ['i', '1', 'I'],
    'I': ['l', '1', 'i'],
    '1': ['l', 'I', 'i'],
    'i': ['l', '1', 'I'],
    '0': ['o', 'O', 'ơ', 'ô'],
    'o': ['0', 'O', 'ơ', 'ô'],
    'O': ['0', 'o', 'Ơ', 'Ô'],
    'a': ['ă', 'â', 'à', 'á', 'ả', 'ã', 'ạ'],
    'ă': ['a', 'â', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ'],
    'â': ['a', 'ă', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
    'e': ['ê', 'è', 'é', 'ẻ', 'ẽ', 'ẹ'],
    'ê': ['e', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
    'o': ['ô', 'ơ', 'ò', 'ó', 'ỏ', 'õ', 'ọ'],
    'ô': ['o', 'ơ', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ'],
    'ơ': ['o', 'ô', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
    'u': ['ư', 'ù', 'ú', 'ủ', 'ũ', 'ụ'],
    'ư': ['u', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
    'á': ['à', 'ả', 'ã', 'ạ', 'a'],
    'à': ['á', 'ả', 'ã', 'ạ', 'a'],
    'ả': ['á', 'à', 'ã', 'ạ', 'a'],
    'ã': ['á', 'à', 'ả', 'ạ', 'a'],
    'ạ': ['á', 'à', 'ả', 'ã', 'a'],
    'd': ['đ'],
    'đ': ['d'],
    'n': ['m', 'r'],
    'm': ['n', 'r'],
    'r': ['n', 'm'],
}

COMMON_OCR_CORRECTIONS = {
    'quán tử': 'quân tử',
    'quy tai': 'quy tài',
    'trì m': 'trì',
    'thiếp í': 'thiếp ý',
    'yếm lé': 'yểm lệ',
    'yếm le': 'yểm lệ',
    'thủ Thân': 'thù Thân',
    'thù Thần': 'thù Thân',
    'hoai tai': 'hoài tài',
    'hoài tai': 'hoài tài',
}
