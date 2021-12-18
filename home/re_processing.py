from underthesea import word_tokenize

_character = [',','!','?','<','>','@','#','$','%','^','&','*','(',')','-','+','[',']','{','}',':','-',':',';','/','+','-','=', '.', '"', '‘', '’']
f = open("home/data/vietnamese-stopwords.txt", "r",encoding='utf-8-sig')
stop_word_list = []
for i in f:
    stop_word_list.append(i.strip())

def train_re_processing(raw_content):
    # Tách từ + chuyển thành chữ thường
    content_token = []
    for i in raw_content:
        content_token.append(word_tokenize(i.lower()))

    # bỏ các ký tự đặc biệt
    content_token_remove_quote = []
    for x in content_token:
        tokens = []
        for y in x:
            if y not in _character and len(y)>1:
                tokens.append(y)
        content_token_remove_quote.append(tokens)
    
    # Loại bỏ stop-word
    clean_content = []
    for sen in content_token_remove_quote:
        tokens = []
        for x in sen:
            if x not in stop_word_list:
                tokens.append(x)
        clean_content.append(tokens)
    
    result = []
    for i in clean_content:
        sen = ''
        sen = ' '.join(i)
        result.append(sen)
    return result

def redict_processing(raw):
    # Tách từ + chuyển thành chữ thường
    tokens = word_tokenize(raw.lower())
    content_token_remove_quote = []

    # bỏ các ký tự đặc biệt
    for i in tokens:
        if i not in _character and len(i)>1:
            content_token_remove_quote.append(i)

    clean_content = []

    # Loại bỏ stop-word
    for i in content_token_remove_quote:
        if i not in stop_word_list:
            clean_content.append(i)
    result = []
    sen = ''
    for i in clean_content:
        a = i+" " 
        sen+=a
    result.append(sen)

    return result