"""Lossless chunk bookkeeping; reject model punctuation that changes the body."""
import re
from difflib import SequenceMatcher
from .core import body


SEGMENT_WIDTH = 240


def segments(text, width=SEGMENT_WIDTH):
    if not text or not text.strip():raise ValueError('请输入需要处理的原文。')
    if width<1:raise ValueError('Segment width must be positive')
    result=[];start=0
    while start<len(text):
        end=min(start+width,len(text))
        if end<len(text):
            for i in range(end-1,max(start+width//2,start),-1):
                if text[i] in '\n。！？；':end=i+1;break
        result.append({'index':len(result),'start':start,'end':end,'original':text[start:end]});start=end
    assert ''.join(x['original'] for x in result)==text
    return result


def _append_content(parts, text, start, end):
    # Leave indentation, trailing spaces and blank lines in the source gaps.
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        parts.append(dict(index=len(parts), start=start, end=end, original=text[start:end]))


def document_segments(task, text):
    if task == 'punctuate':
        return segments(text)
    paragraphs = []
    for line in re.finditer(r'[^\r\n]+', text):
        _append_content(paragraphs, text, line.start(), line.end())
    if not paragraphs:
        raise ValueError('请输入需要处理的原文。')
    return paragraphs


def translation_pieces(paragraph):
    """Bound each inference call. Offsets are relative to its visible paragraph."""
    parts = []
    start = 0
    while start < len(paragraph):
        end = min(start + SEGMENT_WIDTH, len(paragraph))
        if end < len(paragraph):
            for index in range(end - 1, start - 1, -1):
                if paragraph[index] in '。！？；!?;':
                    end = index + 1
                    break
        _append_content(parts, paragraph, start, end)
        start = end
    return parts


def review(task,original,prediction):
    valid=bool(prediction.strip()) and (task!='punctuate' or body(original)==body(prediction))
    differences=[]
    if task=='punctuate' and not valid:
        source,candidate=body(original),body(prediction)
        differences=[{'operation':op,'start':a,'end':b,'source':source[a:b],'candidate':candidate[c:d]}
            for op,a,b,c,d in SequenceMatcher(None,source,candidate,autojunk=False).get_opcodes() if op!='equal']
    return {'candidate':prediction,'accepted':valid if task=='punctuate' else False,
            'body_differences':differences,
            'body_preserved':body(original)==body(prediction) if task=='punctuate' else None,
            'text':prediction if valid else original,
            'status':'accepted' if task=='punctuate' and valid else 'review_required',
            'note':('正文一致，标点仍可人工调整。' if valid else '模型改动了正文，已保留原文，请复核候选。') if task=='punctuate' else '模型译文，原意、专名和否定关系仍需人工复核。'}
