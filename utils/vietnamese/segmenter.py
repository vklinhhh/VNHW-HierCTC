import unicodedata


class SimpleVietnameseSegmenter:
    COMPOUND_WORDS = {
        'việt nam', 'hà nội', 'sài gòn', 'đà nẵng', 'hồ chí minh',
        'thành phố', 'quốc gia', 'đất nước', 'con người', 'cuộc sống',
        'gia đình', 'bạn bè', 'anh em', 'cha mẹ', 'vợ chồng',
        'học sinh', 'sinh viên', 'giáo viên', 'bác sĩ', 'kỹ sư',
        'công ty', 'nhà máy', 'trường học', 'bệnh viện', 'ngân hàng',
        'làm việc', 'học tập', 'nghiên cứu', 'phát triển', 'xây dựng',
        'sản xuất', 'kinh doanh', 'quản lý', 'giảng dạy', 'chăm sóc',
        'xinh đẹp', 'thông minh', 'chăm chỉ', 'tốt đẹp', 'hạnh phúc',
        'vui vẻ', 'buồn bã', 'mạnh mẽ', 'yếu đuối', 'giàu có',
        'quân tử', 'tiểu nhân', 'đại nhân', 'hiền nhân', 'thánh nhân',
        'nhân nghĩa', 'lễ nghĩa', 'đạo đức', 'luân lý', 'triết học',
        'hôm nay', 'hôm qua', 'ngày mai', 'tuần này', 'tháng này',
        'năm nay', 'sáng nay', 'chiều nay', 'tối nay', 'đêm nay',
    }
    
    def __init__(self):
        self._build_compound_lookup()
    
    def _build_compound_lookup(self):
        self.compound_first_words = {}
        for compound in self.COMPOUND_WORDS:
            words = compound.split()
            if words:
                first = words[0]
                if first not in self.compound_first_words:
                    self.compound_first_words[first] = []
                self.compound_first_words[first].append(compound)
    
    def segment(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        
        tokens = text.split()
        result = []
        i = 0
        
        while i < len(tokens):
            token_lower = tokens[i].lower()
            matched = False
            
            if token_lower in self.compound_first_words:
                for compound in self.compound_first_words[token_lower]:
                    compound_words = compound.split()
                    compound_len = len(compound_words)
                    
                    if i + compound_len <= len(tokens):
                        candidate = ' '.join(t.lower() for t in tokens[i:i+compound_len])
                        if candidate == compound:
                            result.append('_'.join(tokens[i:i+compound_len]))
                            i += compound_len
                            matched = True
                            break
            
            if not matched:
                result.append(tokens[i])
                i += 1
        
        return ' '.join(result)
    
    def unsegment(self, text: str) -> str:
        return text.replace('_', ' ')
