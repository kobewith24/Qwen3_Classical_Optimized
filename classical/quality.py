"""Auditable text eligibility rules; they are not semantic correctness scores."""
import re
import unicodedata
from .core import body,digest,is_punctuation

FUNCTION_CHARS=set('之乎者也其而于於为為以乃则則所者矣焉哉曰云夫且盖蓋故及与與或是有无無不非可此彼我吾尔爾汝的了在和就都把被很从從中上下一一个各')


def pair_key(item):return digest(sorted((body(item['input']),body(item['reference']))))


def corruption_flags(item):
    text=item['input']+item['reference'];flags=[]
    if any(unicodedata.category(c) in {'Co','Cs'} for c in text):flags.append('private_or_surrogate_character')
    if any(c in text for c in ('□','�','\x00')):flags.append('unresolved_character_placeholder')
    if re.search(r'[A-Za-z]',text):flags.append('latin_or_markup_in_chinese_corpus')
    return flags


def translation_flags(item):
    flags=corruption_flags(item)
    classical=body(item['input'] if item['task']=='c2m' else item['reference'])
    modern=body(item['reference'] if item['task']=='c2m' else item['input'])
    if classical==modern:flags.append('identity_translation_low_information')
    if len(classical)<8 or len(modern)<10:flags.append('translation_too_short_for_this_recipe')
    ratio=len(classical)/max(len(modern),1)
    if not .18<=ratio<=1.6:flags.append('extreme_classical_modern_length_ratio')
    content=set(classical)-FUNCTION_CHARS
    if len(content)>=10 and len(content.intersection(modern))/len(content)<.30:flags.append('low_lexical_alignment_candidate')
    dates=re.findall(r'[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]',classical)
    if dates and not any(d in modern for d in dates) and re.search(r'[0-9一二三四五六七八九十廿卅]+日',modern):
        flags.append('calendar_conversion_needs_external_context')
    return flags


def punctuation_windows(item,target=80,minimum=40,maximum=120):
    """Slice the gold text without adding marks; offsets refer to NFC body chars."""
    if body(item['input'])!=body(item['reference']):raise ValueError('Punctuation body mismatch')
    text=unicodedata.normalize('NFC',item['reference']);spans=[];start=0;n=0;soft=None
    for index,char in enumerate(text):
        if not char.isspace() and not is_punctuation(char):n+=1
        if char in '，、：' and n>=minimum:soft=index+1
        if (char in '。！？；' and n>=target) or n>=maximum:
            end=index+1 if char in '。！？；' or soft is None else soft
            spans.append((start,end));start=end;n=len(body(text[start:index+1]));soft=None
    if start<len(text):
        if spans and len(body(text[start:]))<minimum:spans[-1]=(spans[-1][0],len(text))
        else:spans.append((start,len(text)))
    assert ''.join(text[a:b] for a,b in spans)==text
    children=[];offset=0
    for a,b in spans:
        reference=text[a:b].strip();source=body(reference);end=offset+len(source)
        if len(source)>=minimum:
            child={**item,'id':digest([item['id'],offset,end,reference])[:24],
                'input':source,'reference':reference,'parent_id':item['id'],
                'body_span':{'start':offset,'end':end},'gold_character_span':{'start':a,'end':b}}
            assert body(child['reference'])==source
            children.append(child)
        offset=end
    return children
