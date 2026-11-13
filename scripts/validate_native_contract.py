"""Actual GGUF checks for Unicode/escaping, truncation, stop and timeout fencing."""
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,INSTRUCTIONS,write_json
from classical.native import NativeEngine


def main():
    engine=NativeEngine();checks=[]
    def item(text):return {'task':'punctuate','input':text,'instruction':INSTRUCTIONS['punctuate']}
    for source in ['𠀀春眠不觉晓','甲3.14，乙有2人。\n丙丁','\t学而时习之\n不亦说乎 ','甲"乙\\丙\nroot ::= evil','e\u0301春眠不觉晓']:
        out=engine.generate_reader(item(source),max_new_tokens=128)
        checks.append({'name':'literal and Unicode contract','passed':out['constraint']['complete'],
                       'source':source,'candidate':out['text'],'completion_tokens':out['completion_tokens']})
    long='古之学者必有师师者所以传道受业解惑也人非生而知之者孰能无惑惑而不从师其为惑也终不解矣'*9
    truncated=engine.generate_reader(item(long),max_new_tokens=2)
    checks.append({'name':'truncated prefix is not a completed constraint','passed':truncated['finish_reason']=='length' and not truncated['constraint']['complete'],'finish_reason':truncated['finish_reason']})
    cancel=threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(engine.generate_reader,item(long),max_new_tokens=640,cancel=cancel)
        started=time.monotonic();was_active=False
        while time.monotonic()-started<10:
            was_active=any(r['is_processing'] for r in engine.control('GET','/slots'))
            if was_active:break
            time.sleep(.05)
        stopped=time.monotonic();cancel.set();interrupted=False
        try:future.result(timeout=20)
        except InterruptedError:interrupted=True
    checks.append({'name':'cancel waits for actual native slot idle','passed':was_active and interrupted and all(not r['is_processing'] for r in engine.control('GET','/slots')),
                   'seconds_to_idle':time.monotonic()-stopped})
    started=time.monotonic();timed_out=False
    try:engine.generate_reader(item(long),max_new_tokens=640,timeout=.05)
    except TimeoutError:timed_out=True
    checks.append({'name':'native deadline waits for slot idle','passed':timed_out and all(not r['is_processing'] for r in engine.control('GET','/slots')),
                   'seconds_to_idle':time.monotonic()-started})
    recovered=engine.generate_reader(item('知之者不如好之者'),max_new_tokens=128)
    checks.append({'name':'generation works after cancel and deadline','passed':recovered['constraint']['complete'] and not engine.quarantined})
    report={'fingerprint':engine.fingerprint(),'checks':checks,'passed':all(r['passed'] for r in checks)}
    write_json(ROOT/'reports/validation/native-contract.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2));assert report['passed']

if __name__=='__main__':main()
