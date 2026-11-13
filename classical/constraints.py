"""Source-conditioned GBNF: NFC body and whitespace are immutable, punctuation bounded."""
import json
import unicodedata
from dataclasses import dataclass
from .core import digest,is_punctuation

POLICY_VERSION='source-punctuation-v1'
MARKS='，。！？；：、“”‘’（）《》「」『』—…'


def skeleton(text):
    return ''.join(c for c in unicodedata.normalize('NFC',text) if not is_punctuation(c))


@dataclass(frozen=True)
class PunctuationConstraint:
    source:str
    grammar:str
    fingerprint:str

    def accepts(self,text):
        # Independent postcondition; no repairing a generated candidate into a success.
        text=unicodedata.normalize('NFC',text);states={(0,0)}
        for char in text:
            following=set()
            for position,marks in states:
                if position<len(self.source) and char==self.source[position]:following.add((position+1,0))
                if char in MARKS and marks<2:following.add((position,marks+1))
            states=following
            if not states:return False
        return any(position==len(self.source) for position,_ in states)


def compile_punctuation(source):
    if not isinstance(source,str) or not source.strip():raise ValueError('请输入原文。')
    if len(source)>1024:raise ValueError('约束断句单段超过 1024 字符，请分段。')
    if any(unicodedata.category(c) in {'Cs','Co'} or (unicodedata.category(c)=='Cc' and c not in '\n\r\t') for c in source):
        raise ValueError('原文包含无法可靠处理的控制字符或私用字符，请先校对。')
    fixed=unicodedata.normalize('NFC',source)
    if not skeleton(fixed).strip():raise ValueError('原文没有可断句的正文。')
    # Each literal is escaped by JSON, never interpreted as a grammar rule.
    parts=['mark']
    for char in fixed:parts.extend([json.dumps(char,ensure_ascii=False),'mark'])
    grammar='root ::= '+' '.join(parts)+'\nmark ::= ['+MARKS+']{0,2}\n'
    return PunctuationConstraint(fixed,grammar,digest({'policy':POLICY_VERSION,'grammar':grammar}))
