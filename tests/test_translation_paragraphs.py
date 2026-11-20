import asyncio
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path

import httpx

from api.server import create_app
from classical.core import digest
from classical.documents import DocumentStore


LOTUS = (
    '水陆草木之花，可爱者甚蕃。晋陶渊明独爱菊。自李唐来，世人甚爱牡丹。'
    '予独爱莲之出淤泥而不染，濯清涟而不妖，中通外直，不蔓不枝，香远益清，亭亭净植，可远观而不可亵玩焉。',
    '予谓菊，花之隐逸者也；牡丹，花之富贵者也；莲，花之君子者也。噫！菊之爱，陶后鲜有闻。'
    '莲之爱，同予者何人？牡丹之爱，宜乎众矣。',
)


class TranslationEngine:
    def __init__(self):
        self.inputs = []
        self.block_on = None
        self.blocked = threading.Event()
        self.outputs = {}

    def fingerprint(self):
        return 'paragraph-test-model'

    def generate_reader(self, item, *, cancel, **kwargs):
        source = item['input']
        self.inputs.append((item['task'], source))
        if source == self.block_on:
            self.blocked.set()
            if not cancel.wait(4):
                raise RuntimeError('test timed out waiting for cancellation')
            raise InterruptedError()
        text, reason = self.outputs.get(source, ('译' + source, 'stop'))
        return dict(text=text, finish_reason=reason, prompt_tokens=3, completion_tokens=2)


class TranslationParagraphTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'reader.sqlite'
        self.open_service()

    def open_service(self):
        self.engine = TranslationEngine()
        self.store = DocumentStore(self.path)
        app = create_app(self.engine, 'test', store=self.store)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://reader',
                                        headers={'Authorization': 'Bearer test'})
        self.addAsyncCleanup(self.client.aclose)

    async def process(self, text, task='c2m', **extra):
        response = await self.client.post('/api/process', json=dict(task=task, text=text, **extra))
        response.raise_for_status()
        return response.json()

    async def test_one_submission_translates_each_lotus_paragraph_in_both_directions(self):
        for task in ('c2m', 'm2c'):
            for eol in ('\n', '\r\n', '\r'):
                with self.subTest(task=task, eol=eol):
                    self.engine.inputs.clear()
                    source = eol.join(LOTUS)
                    data = await self.process(source, task)
                    self.assertEqual(self.engine.inputs, [(task, paragraph) for paragraph in LOTUS])
                    self.assertEqual([row['original'] for row in data['segments']], list(LOTUS))
                    self.assertEqual(data['text'], eol.join('译' + p for p in LOTUS))
                    self.assertTrue(data['complete'])
                    self.assertEqual((data['completed_segments'], data['total_segments']), (2, 2))

    async def test_layout_survives_edit_cache_and_reload(self):
        source = '\r\n　𠀀甲 \r\n \t\n\t乙　\r'
        data = await self.process(source)
        self.assertEqual(self.engine.inputs, [('c2m', '𠀀甲'), ('c2m', '乙')])
        self.assertEqual(data['text'], source.replace('𠀀甲', '译𠀀甲').replace('乙', '译乙'))
        endpoint = '/api/documents/' + data['document_id'] + '/segments/1'
        payload = {'text': '第二段校订', 'revision': 0}
        self.assertEqual((await self.client.patch(endpoint, json=payload)).status_code, 200)
        self.assertEqual((await self.client.patch(endpoint, json=payload)).status_code, 409)
        self.open_service()
        resumed = await self.process(source)
        self.assertEqual(self.engine.inputs, [])
        self.assertEqual(resumed['text'], '\r\n　译𠀀甲 \r\n \t\n\t第二段校订　\r')
        self.assertEqual(resumed['text'], self.store.load(data['document_id'])['text'])

    async def test_cancel_in_middle_of_long_paragraph_then_resume_missing_pieces(self):
        first, middle, last = '甲' * 189 + '。', '乙' * 179 + '！', '丙' * 100
        paragraph = first + middle + last
        source = '首段\n' + paragraph
        key = (await self.client.post('/api/documents', json={'task': 'c2m', 'text': source})).json()['document_id']
        self.engine.block_on = middle
        job = 'lotus-cancel'
        request = asyncio.create_task(self.client.post('/api/process', json=dict(task='c2m', text=source, request_id=job)))
        self.assertTrue(await asyncio.to_thread(self.engine.blocked.wait, 3))
        progress = (await self.client.get('/api/progress/' + job)).json()
        self.assertEqual((progress['completed_segments'], progress['total_segments']), (1, 2))
        await self.client.post('/api/cancel', json={'request_id': job})
        self.assertEqual((await request).status_code, 409)
        partial = self.store.load(key)
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['text'], '译首段\n译' + first + middle + last)
        self.open_service()
        resumed = await self.process(source)
        self.assertEqual(self.engine.inputs, [('c2m', middle), ('c2m', last)])
        self.assertTrue(resumed['complete'])
        self.assertEqual(resumed['total_segments'], 2)
        self.assertEqual(resumed['text'], '译首段\n译' + first + '译' + middle + '译' + last)
        self.assertEqual(resumed['segments'][1]['usage']['completion_tokens'], 4)
        self.assertEqual(resumed['segments'][1]['generation_usage']['completion_tokens'], 6)

    async def test_retry_only_missing_translation_and_keep_one_card_without_newline(self):
        last = '乙'
        for first, output in [('甲' * 240, ('', 'stop')), ('丁' * 240, ('半', 'length'))]:
            with self.subTest(output=output):
                source = first + last
                self.engine.outputs[first] = output
                data = await self.process(source)
                self.assertEqual((data['completed_segments'], data['total_segments']), (0, 1))
                self.assertFalse(data['complete'])
                self.engine.inputs.clear()
                del self.engine.outputs[first]
                complete = await self.process(source)
                self.assertEqual(self.engine.inputs, [('c2m', first)])
                self.assertTrue(complete['complete'])
                self.assertEqual(complete['text'], '译' + first + '译' + last)

    async def test_old_translation_is_readable_and_its_cache_is_not_reused(self):
        source = '\n'.join(LOTUS)
        old = digest(dict(task='c2m', original=source, signature='same', width=240))
        with closing(self.store.connect()) as db, db:
            db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)', (old, 'c2m', source, 'old', 1, 0))
        self.store.save_generated(old, dict(index=0, start=0, end=len(source), original=source,
                                            text='旧译文', status='review_required'))
        self.store.edit(old, 0, '旧校订', 0)
        new = self.store.prepare('c2m', source, 'same', 'old')
        self.assertNotEqual(new, old)
        self.assertEqual(self.store.load(old)['text'], '旧校订')
        self.assertEqual(self.store.load(new)['total_segments'], 2)
        self.assertFalse(self.store.load(new)['complete'])
