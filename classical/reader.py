"""Lossless chunk bookkeeping; reject model punctuation that changes the body."""
from difflib import SequenceMatcher
from .core import body


def segments(text, width=240):
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
