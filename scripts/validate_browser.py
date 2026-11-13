"""Isolated headless Edge E2E against the actual gateway, SQLite and GPU backend."""
import json
import re
import sys
import uuid
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from classical.core import ROOT,file_digest,write_json


def main():
    folder=ROOT/'reports/validation';checks=[];failure=None
    source='学而时习之不亦说乎';edited='“学而时习之，不亦说乎？”'
    def passed(name,detail=None):checks.append({'name':name,'passed':True,'observed':detail})
    try:
        with sync_playwright() as p:
            context=p.chromium.launch_persistent_context(str(ROOT/'cache'/('ui-profile-'+uuid.uuid4().hex)),
                executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True,
                viewport={'width':1366,'height':900},args=['--disable-gpu','--disable-background-networking'])
            try:
                page=context.pages[0] if context.pages else context.new_page()
                page.goto('http://127.0.0.1:8010',wait_until='domcontentloaded')
                page.locator('#token').fill((ROOT/'data/access.token').read_text().strip())
                page.locator('#access-form button').click()
                expect(page.locator('#model-name')).to_contain_text('clean_2k_seed24')
                passed('real page connects to default model')
                page.locator('#source').fill(source);page.locator('#run').click()
                expect(page.locator('#status')).to_have_text('完成 · 正文一致',timeout=60000)
                page.locator('#result summary').first.click()
                editor=page.locator('.review-editor').first;save=page.get_by_role('button',name='保存校对',exact=True).first
                editor.fill(source+'也');save.click()
                expect(page.locator('.review-feedback').first).to_contain_text('正文仍有改字或漏字')
                passed('UI rejects edited body')
                editor.fill(edited);save.click()
                expect(page.locator('.review-feedback').first).to_contain_text('已保存到本机')
                passed('UI saves corrected punctuation to server')
                page.reload(wait_until='domcontentloaded')
                expect(page.locator('#status')).to_contain_text('已恢复 1/1',timeout=20000)
                expect(page.locator('#source')).to_have_value(source)
                expect(page.locator('#result .segment > p').first).to_have_text(edited)
                passed('source and manual revision survive page refresh')
                short_id=page.locator('#history').input_value()
                page.locator('#source').fill('知之者不如好之者')
                page.locator('#history').select_option('');page.locator('#history').select_option(short_id)
                expect(page.locator('#source')).to_have_value(source)
                expect(page.locator('#result .segment > p').first).to_have_text(edited)
                passed('history restores source task and manual result')
                long='君子曰学不可以已青取之于蓝而青于蓝冰水为之而寒于水木直中绳輮以为轮其曲中规虽有槁暴不复挺者輮使之然也故木受绳则直金就砺则利君子博学而日参省乎己则知明而行无过矣吾尝终日而思矣不如须臾之所学也吾尝跂而望矣不如登高之博见也登高而招臂非加长也而见者远顺风而呼声非加疾也而闻者彰\n'+'积土成山风雨兴焉积水成渊蛟龙生焉积善成德而神明自得圣心备焉'*18+'\n【页面验收'+uuid.uuid4().hex[:8]+'】'
                page.locator('#source').fill(long);page.locator('#run').click()
                expect(page.locator('#status')).to_have_text(re.compile(r'已完成 [1-9]\d*/\d+ 段.*'),timeout=60000)
                page.locator('#cancel').click()
                expect(page.locator('#status')).to_contain_text('已停止',timeout=20000)
                expect(page.locator('#source')).to_have_value(long)
                expect(page.locator('#run')).to_be_enabled()
                expect(page.locator('#result .segment').first).to_be_visible()
                passed('cancel shows saved partial results and keeps source')
                page.screenshot(path=str(folder/'reader-partial.png'),full_page=True)
                page.reload(wait_until='domcontentloaded')
                expect(page.locator('#status')).to_have_text(re.compile(r'已恢复 [1-9]\d*/\d+ 段.*'),timeout=20000)
                expect(page.locator('#source')).to_have_value(long)
                passed('partial document is restored after refresh')
                page.locator('#run').click()
                expect(page.locator('#status')).to_have_text('完成 · 正文一致',timeout=150000)
                expect(page.locator('#export')).to_be_enabled()
                passed('UI resumes remaining work and enables export')
                page.locator('#history').select_option(short_id)
                expect(page.locator('#source')).to_have_value(source)
                expect(page.locator('#result .segment > p').first).to_have_text(edited)
                page.screenshot(path=str(folder/'reader-resume.png'),full_page=True)
                passed('previous manual correction remains intact after another document')
            finally:context.close()
    except Exception as exc:
        failure=type(exc).__name__+': '+str(exc)
    report={'method':'isolated headless Edge via existing Playwright; actual DOM controls, actual authenticated gateway, SQLite and native GPU; no mocks',
            'desktop_tool_note':'CUA and node_repl initialization failed with kernel-assets path error; no desktop-UI success claimed',
            'passed':failure is None,'checks':checks,'error':failure,'screenshot':'reader-resume.png',
            'code_sha256':{name:file_digest(ROOT/name) for name in ['api/server.py','api/static/app.js','api/static/index.html','api/static/app.css','scripts/validate_browser.py']}}
    write_json(folder/'browser-validation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    assert report['passed'],failure

if __name__=='__main__':main()
