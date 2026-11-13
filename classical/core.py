import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = '你是一位精通文言文的学者。准确、简洁地完成用户要求，不添加解释。断句时只能添加标点，不得改动、增删或调换原文字词。'
INSTRUCTIONS = {
    'punctuate': '给下列古文添加标点，只输出断句后的原文，不得改动、增删或调换任何原字：',
    'c2m': '将下列古文翻译成现代汉语，只输出译文，保留原意、人名和地名，不添加原文没有的信息：',
    'm2c': '将下列现代汉语翻译成文言文，只输出译文，保留原意、人名和地名：',
}


def digest(value):
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{n}: invalid JSON') from e


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def is_punctuation(char):
    return unicodedata.category(char).startswith('P') or char in '—…·'


def body(text):
    return ''.join(c for c in unicodedata.normalize('NFC', text or '') if not c.isspace() and not is_punctuation(c))


def punctuation_positions(text, typed=False):
    positions = set(); n = 0
    for char in unicodedata.normalize('NFC', text or ''):
        if is_punctuation(char): positions.add((n, char) if typed else n)
        elif not char.isspace(): n += 1
    return positions


def set_f1(predicted, reference):
    if not predicted and not reference: return 1.0
    return 2 * len(predicted & reference) / (len(predicted) + len(reference))


def edit_distance(a,b):
    if a==b:return 0
    start=0
    while start<min(len(a),len(b)) and a[start]==b[start]:start+=1
    a=a[start:];b=b[start:]
    while a and b and a[-1]==b[-1]:a=a[:-1];b=b[:-1]
    if not a:return len(b)
    if not b:return len(a)
    if len(a)<len(b):a,b=b,a
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]


def punctuation_scores(source, prediction, reference):
    if body(source) != body(reference):
        raise ValueError('Invalid punctuation reference: body differs from source')
    preserved = body(source) == body(prediction)
    boundary = set_f1(punctuation_positions(prediction), punctuation_positions(reference))
    typed = set_f1(punctuation_positions(prediction, True), punctuation_positions(reference, True))
    return {'score': boundary if preserved else 0.0, 'boundary_f1': boundary,
            'body_character_error_rate':edit_distance(body(source),body(prediction))/max(len(body(source)),1),
            'typed_f1': typed if preserved else 0.0, 'body_preserved': preserved}


def score(item, prediction):
    task, reference = item['task'], item['reference']
    if task == 'punctuate': return punctuation_scores(item['input'], prediction, reference)
    if task == 'fill_in': return {'score': float(bool(body(prediction)) and body(prediction) == body(reference))}
    if task in {'idiom_source', 'source'}:
        book = item.get('metadata', {}).get('book', '').strip('《》 ')
        if not book: raise ValueError('Source scoring requires a reference book')
        return {'score': float(book in prediction.replace('《', '').replace('》', ''))}
    # Missing dependencies must fail, never silently change the metric.
    from sacrebleu.metrics import CHRF
    return {'score': CHRF(word_order=0).sentence_score(prediction.strip(), [reference.strip()]).score / 100}


def messages(item):
    instruction = item.get('instruction') or INSTRUCTIONS.get(item['task'], '')
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': (instruction + '\n' + item['input']).strip()}]


def tokenize_record(tokenizer, item, max_length=1024):
    turns = messages(item)
    prompt = tokenizer.apply_chat_template(turns, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    complete = tokenizer.apply_chat_template(turns + [{'role': 'assistant', 'content': item['reference']}],
                                            tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if hasattr(prompt, 'get'): prompt = prompt['input_ids']
    if hasattr(complete, 'get'): complete = complete['input_ids']
    if complete[:len(prompt)] != prompt: raise ValueError('Chat template boundary is not prefix aligned')
    if len(complete) > max_length: raise OverflowError('Sample exceeds token budget; not silently truncated')
    labels = [-100] * len(prompt) + complete[len(prompt):]
    if not any(x != -100 for x in labels): raise ValueError('No supervised answer tokens')
    return {'input_ids': complete, 'attention_mask': [1] * len(complete), 'labels': labels,
            'task': item['task'], 'book': item.get('book', ''), 'sample_id': item['id'],
            'supervised_tokens': len(complete) - len(prompt), 'prompt_tokens': len(prompt)}
