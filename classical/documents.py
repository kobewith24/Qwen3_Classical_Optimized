"""Local resumable document results; exact model/policy identity and revisioned edits."""
import json
import sqlite3
import time
from contextlib import closing
from .core import body,digest
from .reader import segments


def assemble(original,rows):
    chunks=[];offset=0
    for row in sorted(rows,key=lambda r:r['start']):
        chunks.extend([original[offset:row['start']],row['text']]);offset=row['end']
    return ''.join(chunks)+original[offset:]


class DocumentStore:
    def __init__(self,path):
        self.path=path;path.parent.mkdir(parents=True,exist_ok=True)
        with closing(self.connect()) as db,db:
            db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, task TEXT, original TEXT, model TEXT, total INTEGER, created REAL);
            CREATE TABLE IF NOT EXISTS chunks(document TEXT, idx INTEGER, result TEXT, revision INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(document,idx),FOREIGN KEY(document) REFERENCES documents(id));''')

    def connect(self):
        db=sqlite3.connect(self.path,timeout=5);db.execute('PRAGMA foreign_keys=ON');return db

    def prepare(self,task,text,signature,model):
        parts=segments(text);key=digest({'task':task,'original':text,'signature':signature,'width':240})
        with closing(self.connect()) as db,db:
            db.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?)',(key,task,text,model,len(parts),time.time()))
        return key

    def load(self,key):
        with closing(self.connect()) as db:
            row=db.execute('SELECT task,original,model,total FROM documents WHERE id=?',(key,)).fetchone()
            if row is None:raise KeyError(key)
            rows=[{**json.loads(value),'revision':revision} for value,revision in db.execute(
                'SELECT result,revision FROM chunks WHERE document=? ORDER BY idx',(key,))]
        task,original,model,total=row
        completed=sum(r.get('finish_reason','stop')=='stop' or r['status']=='user_confirmed' for r in rows)
        return {'document_id':key,'task':task,'original':original,'model':model,'segments':rows,
                'completed_segments':completed,'total_segments':total,'complete':completed==total,
                'review_required':sum(r['status']=='review_required' for r in rows),'text':assemble(original,rows)}

    def recent(self):
        with closing(self.connect()) as db:
            keys=[r[0] for r in db.execute('SELECT id FROM documents ORDER BY created DESC LIMIT 20')]
        result=[]
        for key in keys:
            data=self.load(key)
            result.append({k:data[k] for k in ['document_id','task','model','completed_segments','total_segments']}|
                          {'preview':data['original'][:36].replace('\n',' ')})
        return result

    def save_generated(self,key,row):
        with closing(self.connect()) as db,db:
            db.execute('INSERT INTO chunks(document,idx,result) VALUES(?,?,?) ON CONFLICT(document,idx) DO UPDATE SET result=excluded.result WHERE chunks.revision=0',
                       (key,row['index'],json.dumps(row,ensure_ascii=False)))
        return {**row,'revision':0}

    def edit(self,key,index,text,revision):
        with closing(self.connect()) as db,db:
            row=db.execute('SELECT chunks.result,documents.task FROM chunks JOIN documents ON documents.id=chunks.document WHERE document=? AND idx=?',
                           (key,index)).fetchone()
            if row is None:raise KeyError((key,index))
            result=json.loads(row[0])
            if not text.strip():raise ValueError('校对文本不能为空。')
            if row[1]=='punctuate' and body(text)!=body(result['original']):raise ValueError('正文仍有改字或漏字。')
            result.update(text=text,status='user_confirmed',note='已保存你的校对。')
            updated=db.execute('UPDATE chunks SET result=?,revision=revision+1 WHERE document=? AND idx=? AND revision=?',
                               (json.dumps(result,ensure_ascii=False),key,index,revision)).rowcount
            if not updated:raise RuntimeError('该段已在其他页面更新，请刷新后再保存。')
        return {**result,'revision':revision+1}
