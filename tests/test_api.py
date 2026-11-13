import asyncio
import json
import queue
import time
import unittest

import httpx
from api.server import create_app


class FakeStreamer:
    def __init__(self):self.queue=queue.Queue()
    def __iter__(self):return self
    def __next__(self):
        value=self.queue.get(timeout=.02)
        if value is None:raise StopIteration
        return value


class Engine:
    def __init__(self,text='春眠，不觉。',delay=.01,error=None):self.text=text;self.delay=delay;self.error=error;self.calls=0;self.cancelled=False
    def make_streamer(self):return FakeStreamer()
    def generate_messages(self,turns,max_new_tokens,cancel,timeout,streamer=None):
        self.calls+=1;start=time.monotonic()
        while time.monotonic()-start<self.delay:
            if cancel.is_set():self.cancelled=True;raise InterruptedError()
            time.sleep(.002)
        if self.error:raise self.error
        if streamer:
            streamer.queue.put(self.text[:2]);streamer.queue.put(self.text[2:]);streamer.queue.put(None)
        return {'text':self.text,'prompt_tokens':11,'completion_tokens':7,'finish_reason':'stop','batch_seconds':self.delay}


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine=Engine();self.app=create_app(self.engine,'test-secret',queue_timeout=.025)
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://test')
        self.headers={'Authorization':'Bearer test-secret'}
        self.payload={'messages':[{'role':'user','content':'断句：春眠不觉'}]}

    async def asyncTearDown(self):await self.client.aclose()

    async def test_auth_is_required_for_chat_process_and_metrics(self):
        self.assertEqual((await self.client.post('/v1/chat/completions',json=self.payload)).status_code,401)
        self.assertEqual((await self.client.post('/api/process',json={'task':'punctuate','text':'春眠不觉'})).status_code,401)
        self.assertEqual((await self.client.get('/metrics')).status_code,401)
        self.assertEqual((await self.client.post('/api/cancel',json={'request_id':'cancel-test-24'})).status_code,401)
        self.assertEqual((await self.client.get('/api/progress/progress-test-24')).status_code,401)
        self.assertEqual(self.engine.calls,0)

    async def test_chat_usage_is_actual_token_count(self):
        result=await self.client.post('/v1/chat/completions',headers=self.headers,json=self.payload)
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.json()['usage']['completion_tokens'],7)
        self.assertEqual(self.app.state.gate.active,0)

    async def test_sse_has_content_done_and_usage_independent_of_chunk_count(self):
        result=await self.client.post('/v1/chat/completions',headers=self.headers,
            json={**self.payload,'stream':True,'stream_options':{'include_usage':True}})
        data=[line[6:] for line in result.text.splitlines() if line.startswith('data: ')]
        self.assertEqual(data[-1],'[DONE]')
        chunks=[json.loads(x) for x in data[:-1]]
        self.assertEqual(''.join(x.get('choices',[{}])[0].get('delta',{}).get('content','') for x in chunks if x.get('choices')),'春眠，不觉。')
        self.assertEqual(next(x['usage']['completion_tokens'] for x in chunks if 'usage' in x),7)

    async def test_queue_timeout_does_not_leak_capacity(self):
        await self.app.state.gate.acquire()
        try:
            result=await self.client.post('/v1/chat/completions',headers=self.headers,json=self.payload)
            self.assertEqual(result.status_code,429)
            self.assertEqual(self.app.state.gate.outstanding,1)
        finally:self.app.state.gate.release()

    async def test_full_queue_rejects_without_starting_generation(self):
        self.app.state.gate.capacity=0;await self.app.state.gate.acquire()
        try:
            result=await self.client.post('/v1/chat/completions',headers=self.headers,json=self.payload)
            self.assertEqual(result.status_code,429);self.assertEqual(self.engine.calls,0)
        finally:self.app.state.gate.release()

    async def test_timeout_releases_actual_worker_slot(self):
        self.engine.error=TimeoutError()
        result=await self.client.post('/v1/chat/completions',headers=self.headers,json=self.payload)
        self.assertEqual(result.status_code,504);await asyncio.sleep(.01)
        self.assertEqual(self.app.state.gate.active,0)

    async def test_cancelled_client_signals_worker_and_releases_slot(self):
        self.engine.delay=1
        request=asyncio.create_task(self.client.post('/v1/chat/completions',headers=self.headers,json=self.payload))
        while not self.engine.calls:await asyncio.sleep(.005)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):await request
        for _ in range(100):
            if not self.app.state.gate.active:break
            await asyncio.sleep(.005)
        self.assertTrue(self.engine.cancelled);self.assertEqual(self.app.state.gate.active,0)

    async def test_reader_rejects_changed_original_characters(self):
        self.engine.text='甲乙，丙丁。'
        result=await self.client.post('/api/process',headers=self.headers,json={'task':'punctuate','text':'春眠不觉'})
        self.assertEqual(result.status_code,200);data=result.json()
        self.assertEqual(data['text'],'春眠不觉');self.assertEqual(data['review_required'],1)

    async def test_explicit_cancel_stops_reader_worker_before_next_request(self):
        self.engine.delay=2;key='reader-cancel-24'
        request=asyncio.create_task(self.client.post('/api/process',headers=self.headers,
            json={'task':'punctuate','text':'春眠不觉','request_id':key}))
        while not self.engine.calls:await asyncio.sleep(.005)
        cancelled=await self.client.post('/api/cancel',headers=self.headers,json={'request_id':key})
        self.assertEqual(cancelled.status_code,200)
        self.assertEqual((await request).status_code,409)
        for _ in range(100):
            if self.app.state.gate.active==0:break
            await asyncio.sleep(.005)
        self.assertTrue(self.engine.cancelled);self.assertEqual(self.app.state.gate.active,0)
        self.assertEqual(len(self.app.state.jobs),0)
        self.engine.delay=.01
        self.assertEqual((await self.client.post('/api/process',headers=self.headers,
            json={'task':'punctuate','text':'春眠不觉','request_id':'next-reader-24'})).status_code,200)

    async def test_cancel_before_delivery_prevents_generation(self):
        key='before-start-24'
        await self.client.post('/api/cancel',headers=self.headers,json={'request_id':key})
        response=await self.client.post('/api/process',headers=self.headers,
            json={'task':'punctuate','text':'春眠不觉','request_id':key})
        self.assertEqual(response.status_code,409);self.assertEqual(self.engine.calls,0)
        self.assertEqual(self.app.state.gate.outstanding,0)

    async def test_cancel_queued_reader_releases_only_its_queue_slot(self):
        self.app.state.gate.timeout=2;await self.app.state.gate.acquire();key='queued-cancel-24'
        try:
            request=asyncio.create_task(self.client.post('/api/process',headers=self.headers,
                json={'task':'punctuate','text':'春眠不觉','request_id':key}))
            while self.app.state.gate.outstanding<2:await asyncio.sleep(.005)
            await self.client.post('/api/cancel',headers=self.headers,json={'request_id':key})
            self.assertEqual((await request).status_code,409)
            self.assertEqual(self.app.state.gate.outstanding,1);self.assertEqual(self.engine.calls,0)
        finally:self.app.state.gate.release()

    async def test_progress_reports_real_completed_segments_without_text(self):
        self.engine.delay=.06;key='progress-test-24'
        request=asyncio.create_task(self.client.post('/api/process',headers=self.headers,
            json={'task':'punctuate','text':'春眠不觉'*130,'request_id':key}))
        observed=None
        for _ in range(100):
            response=await self.client.get('/api/progress/'+key,headers=self.headers)
            if response.status_code==200 and response.json()['completed_segments']>0:
                observed=response.json();break
            await asyncio.sleep(.005)
        self.assertIsNotNone(observed);self.assertEqual(observed['total_segments'],3)
        self.assertLess(observed['completed_segments'],3);self.assertNotIn('text',observed);self.assertNotIn('original',observed)
        self.assertEqual((await request).status_code,200)
        self.assertEqual((await self.client.get('/api/progress/'+key,headers=self.headers)).status_code,404)

    async def test_whole_reader_deadline_does_not_reset_per_segment(self):
        self.engine.delay=.04;app=create_app(self.engine,'test-secret',reader_task_timeout=.07)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://deadline') as client:
            response=await client.post('/api/process',headers=self.headers,
                json={'task':'punctuate','text':'春眠不觉'*130,'request_id':'deadline-test-24'})
            self.assertEqual(response.status_code,504)
            for _ in range(100):
                if app.state.gate.active==0:break
                await asyncio.sleep(.005)
            self.assertEqual(app.state.gate.outstanding,0);self.assertEqual(len(app.state.jobs),0)
            self.assertEqual(len(app.state.progress),0)
            self.assertLess(self.engine.calls,3)
            self.assertEqual((await client.get('/metrics',headers=self.headers)).json()['reader_timeouts'],1)

    async def test_whole_reader_deadline_signals_running_worker(self):
        self.engine.delay=1;app=create_app(self.engine,'test-secret',reader_task_timeout=.2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://deadline') as client:
            response=await client.post('/api/process',headers=self.headers,
                json={'task':'punctuate','text':'春眠不觉','request_id':'running-deadline-24'})
            self.assertEqual(response.status_code,504)
            for _ in range(100):
                if app.state.gate.active==0:break
                await asyncio.sleep(.005)
            self.assertTrue(self.engine.cancelled);self.assertEqual(app.state.gate.outstanding,0)

    async def test_whole_reader_deadline_includes_waiting_queue(self):
        app=create_app(self.engine,'test-secret',reader_task_timeout=.02,queue_timeout=10)
        await app.state.gate.acquire()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://deadline') as client:
                response=await client.post('/api/process',headers=self.headers,
                    json={'task':'punctuate','text':'春眠不觉','request_id':'queued-deadline-24'})
                self.assertEqual(response.status_code,504);self.assertEqual(self.engine.calls,0)
                self.assertEqual(app.state.gate.outstanding,1);self.assertEqual(len(app.state.progress),0)
        finally:app.state.gate.release()

    async def test_unknown_model_and_oversized_input_are_rejected(self):
        result=await self.client.post('/v1/chat/completions',headers=self.headers,json={**self.payload,'model':'other'})
        self.assertEqual(result.status_code,400)
        result=await self.client.post('/api/process',headers=self.headers,json={'task':'punctuate','text':'甲'*3001})
        self.assertEqual(result.status_code,422)


if __name__=='__main__':unittest.main()
