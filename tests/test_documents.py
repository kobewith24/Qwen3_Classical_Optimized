import asyncio
import tempfile
import unittest
from pathlib import Path
import httpx
from api.server import create_app
from classical.documents import DocumentStore,assemble
from test_api import Engine


class StableEngine(Engine):
    def fingerprint(self):return 'stable-test-model'


class DocumentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'reader.sqlite'
        self.store=DocumentStore(self.path);self.engine=StableEngine(delay=.02)
        self.app=create_app(self.engine,'secret',store=self.store)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://test')
        self.headers={'Authorization':'Bearer secret'}

    async def asyncTearDown(self):await self.client.aclose();self.temp.cleanup()

    async def test_identity_depends_on_model_task_and_exact_source(self):
        key=self.store.prepare('punctuate','甲乙','v1','model')
        self.assertEqual(key,self.store.prepare('punctuate','甲乙','v1','model'))
        for task,text,signature in [('c2m','甲乙','v1'),('punctuate','甲乙 ','v1'),('punctuate','甲乙','v2')]:
            self.assertNotEqual(key,self.store.prepare(task,text,signature,'model'))

    async def test_documents_are_private_and_edits_require_matching_body_and_revision(self):
        self.assertEqual((await self.client.post('/api/documents',json={'task':'punctuate','text':'春眠不觉'})).status_code,401)
        generated=await self.client.post('/api/process',headers=self.headers,json={'task':'punctuate','text':'春眠不觉'})
        key=generated.json()['document_id'];endpoint=f'/api/documents/{key}/segments/0'
        self.assertEqual((await self.client.get('/api/documents/'+key)).status_code,401)
        self.assertEqual((await self.client.patch(endpoint,headers=self.headers,json={'text':'春眠不晓。','revision':0})).status_code,400)
        good=await self.client.patch(endpoint,headers=self.headers,json={'text':'春眠，不觉！','revision':0})
        self.assertEqual(good.status_code,200);self.assertEqual(good.json()['revision'],1)
        self.assertEqual((await self.client.patch(endpoint,headers=self.headers,json={'text':'春眠不觉。','revision':0})).status_code,409)
        self.assertEqual(self.store.load(key)['text'],'春眠，不觉！')
        calls=self.engine.calls
        reused=await self.client.post('/api/process',headers=self.headers,json={'task':'punctuate','text':'春眠不觉'})
        row=reused.json()['segments'][0]
        self.assertEqual(self.engine.calls,calls);self.assertTrue(row['cache_hit'])
        self.assertEqual(row['usage']['completion_tokens'],0)
        self.assertGreater(row['generation_usage']['completion_tokens'],0)

    async def test_cancelled_document_resumes_after_restart_without_repeating_saved_segments(self):
        text='春眠不觉'*130;key=(await self.client.post('/api/documents',headers=self.headers,json={'task':'punctuate','text':text})).json()['document_id']
        self.engine.delay=.08
        request=asyncio.create_task(self.client.post('/api/process',headers=self.headers,
            json={'task':'punctuate','text':text,'request_id':'resume-test-24'}))
        while self.store.load(key)['completed_segments']==0:await asyncio.sleep(.005)
        await self.client.post('/api/cancel',headers=self.headers,json={'request_id':'resume-test-24'})
        self.assertEqual((await request).status_code,409)
        while self.app.state.gate.active:await asyncio.sleep(.005)
        saved=self.store.load(key)['completed_segments'];self.assertGreater(saved,0)
        engine=StableEngine();app=create_app(engine,'secret',store=DocumentStore(self.path))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as restarted:
            response=await restarted.post('/api/process',headers=self.headers,json={'task':'punctuate','text':text})
        self.assertTrue(response.json()['complete'])
        self.assertEqual(engine.calls,3-saved)
        self.assertEqual(self.store.load(key)['text'],text)

    async def test_length_truncated_candidate_is_not_saved_as_a_completed_segment(self):
        original=self.engine.generate_messages
        def incomplete(*args,**kwargs):return {**original(*args,**kwargs),'finish_reason':'length'}
        self.engine.generate_messages=incomplete
        response=await self.client.post('/api/process',headers=self.headers,json={'task':'punctuate','text':'春眠不觉'})
        self.assertFalse(response.json()['complete'])
        key=response.json()['document_id']
        self.assertEqual(self.store.load(key)['completed_segments'],0)
        edit=await self.client.patch(f'/api/documents/{key}/segments/0',headers=self.headers,json={'text':'春眠，不觉。','revision':0})
        self.assertEqual(edit.status_code,200)
        self.assertTrue(self.store.load(key)['complete'])
        calls=self.engine.calls
        resumed=await self.client.post('/api/process',headers=self.headers,json={'task':'punctuate','text':'春眠不觉'})
        self.assertTrue(resumed.json()['complete']);self.assertEqual(self.engine.calls,calls)

    def test_partial_assembly_keeps_unprocessed_text_and_layout(self):
        self.assertEqual(assemble('甲乙\n丙丁',[{'start':0,'end':2,'text':'甲，乙。'}]),'甲，乙。\n丙丁')


class NativeRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_constraints_are_only_requested_by_reader_endpoint(self):
        class Native(Engine):
            def __init__(self):super().__init__();self.reader_calls=0
            def generate_reader(self,item,**kwargs):self.reader_calls+=1;return self.generate_messages([],**kwargs)
        engine=Native();app=create_app(engine,'secret')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            headers={'Authorization':'Bearer secret'}
            self.assertEqual((await client.post('/api/process',headers=headers,json={'task':'punctuate','text':'春眠不觉'})).status_code,200)
            self.assertEqual((await client.post('/v1/chat/completions',headers=headers,json={'messages':[{'role':'user','content':'你好'}]})).status_code,200)
        self.assertEqual(engine.reader_calls,1);self.assertEqual(engine.calls,2)

if __name__=='__main__':unittest.main()
