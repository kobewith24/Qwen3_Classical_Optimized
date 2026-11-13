"""Pin the official C-Eval val data and freeze a 5-per-subject diagnostic sample."""
import concurrent.futures
import json
import random
import sys
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,file_digest,write_json,write_jsonl


def main():
    output=ROOT/'data/ceval_val_260'
    if output.exists():raise FileExistsError(output)
    repository=json.loads((ROOT/'reports/2026-11-13/ceval-repository.json').read_text(encoding='utf-8'))
    revision=repository['sha'];files=sorted(r['rfilename'] for r in repository['siblings'] if '/val-' in r['rfilename'])
    if len(files)!=52:raise ValueError('Expected all 52 official subject validation files')
    folder=ROOT/'assets/ceval'/revision;folder.mkdir(parents=True,exist_ok=True)
    mapping_url='https://raw.githubusercontent.com/hkust-nlp/ceval/main/subject_mapping.json'
    mapping_response=requests.get(mapping_url,timeout=(10,30));mapping_response.raise_for_status();mapping=mapping_response.json()
    write_json(folder/'subject_mapping.json',mapping)
    def fetch(name):
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            response=requests.get(f'https://huggingface.co/datasets/ceval/ceval-exam/resolve/{revision}/{name}',timeout=(10,60))
            response.raise_for_status();pending=path.with_suffix('.download');pending.write_bytes(response.content);pending.rename(path)
        if path.read_bytes()[:4]!=b'PAR1':raise ValueError('Invalid parquet '+name)
        return {'file':name,'sha256':file_digest(path),'bytes':path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:downloaded=list(pool.map(fetch,files))
    import pyarrow.parquet as parquet
    rng=random.Random(20261113);chosen=[]
    for name in files:
        subject=name.split('/')[0];rows=parquet.read_table(folder/name).to_pylist()
        if len(rows)<5:raise ValueError('Insufficient val examples')
        rng.shuffle(rows)
        for row in rows[:5]:
            if row['answer'] not in 'ABCD' or len(row['answer'])!=1:raise ValueError('Missing official answer')
            chosen.append({'id':f'ceval/val/{subject}/{row["id"]}','subject':subject,'subject_zh':mapping[subject][1],
                'category':mapping[subject][2],'question':row['question'],'choices':{k:row[k] for k in 'ABCD'},'answer':row['answer']})
    output.mkdir(parents=True);write_jsonl(output/'questions.jsonl',chosen)
    write_json(output/'manifest.json',{'source':'https://huggingface.co/datasets/ceval/ceval-exam','revision':revision,
        'source_instructions':'https://github.com/hkust-nlp/ceval#how-to-evaluate-on-c-eval',
        'license':repository.get('cardData',{}).get('license'),'split':'val','selection_seed':20261113,
        'subjects':52,'samples_per_subject':5,'samples':len(chosen),'files':downloaded,
        'mapping_source':mapping_url,'mapping_sha256':file_digest(folder/'subject_mapping.json'),
        'questions_sha256':file_digest(output/'questions.jsonl'),
        'scope':'260-item stratified validation diagnostic, not the full test or an official leaderboard result; no proof of overall ability retention'})
    print('OFFICIAL_CEVAL_SAMPLE_FROZEN',len(chosen),revision,flush=True)

if __name__=='__main__':main()
