'use strict';
const $=id=>document.getElementById(id);let token=sessionStorage.getItem('classical_token')||'',controller=null,lastResult=null,activeRequestId=null,cancelFeedback='已停止，原文保留。',progressTimer=null,stopping=false;
const auth=()=>({'Authorization':`Bearer ${token}`,'Content-Type':'application/json'});
let documentId=null,viewEpoch=0;
const rememberDocument=id=>{documentId=id;try{if(id)localStorage.setItem('classical_document',id);else localStorage.removeItem('classical_document');}catch{/* Saving on the server still works when browser storage is disabled. */}};
function assembledText(data){let text='',offset=0;for(const row of [...data.segments].sort((a,b)=>a.start-b.start)){text+=data.original.slice(offset,row.start)+row.text;offset=row.end;}return text+data.original.slice(offset);}
async function refreshHistory(){try{const response=await fetch('/api/documents',{headers:auth()});if(!response.ok)return;const data=await response.json();$('history').replaceChildren(new Option('选择本机记录',''));const names={punctuate:'断句',c2m:'古译今',m2c:'今译古'};for(const row of data.documents){$('history').add(new Option(`${names[row.task]} · ${row.completed_segments}/${row.total_segments} · ${row.preview}`,row.document_id));}if(documentId)$('history').value=documentId;}catch{}}
async function loadDocument(id,restoreSource=false){const epoch=++viewEpoch;const response=await fetch('/api/documents/'+id,{headers:auth()});if(!response.ok)throw Error('未能恢复本机文档。');const data=await response.json();if(epoch!==viewEpoch)return null;rememberDocument(id);if(restoreSource){$('source').value=data.original;$('task').value=data.task;$('count').textContent=`${data.original.length} / 3000`;}output(data);return data;}
async function connect(){if(!token){$('access').showModal();return;}const response=await fetch('/v1/models',{headers:auth()});if(!response.ok)throw Error('令牌无效，请检查新项目 data/access.token。');const data=await response.json();$('model-name').textContent=`本地模型 · ${data.data[0].id}`;$('access').close();let previous;try{previous=localStorage.getItem('classical_document');}catch{}if(previous&&!controller&&!$('source').value){try{const restored=await loadDocument(previous,true);if(!restored)return;$('status').textContent=`已恢复 ${restored.completed_segments}/${restored.total_segments} 段 · 点击开始可继续`;}catch{$('status').textContent='上次记录暂不可恢复，请稍后重试。';}}await refreshHistory();}
function output(data){
  lastResult=data;$('result').replaceChildren();$('result').classList.remove('empty');
  if(!data.segments.length){$('result').classList.add('empty');$('result').textContent='原文已保存，点击开始可继续处理。';}
  for(const segment of data.segments){
    const article=document.createElement('article');article.className='segment'+(segment.status==='review_required'?' review':'');
    const label=document.createElement('small');label.textContent=`第 ${segment.index+1} 段 · ${segment.note}`;
    const text=document.createElement('p');text.textContent=segment.text;article.append(label,text);
    const details=document.createElement('details'),summary=document.createElement('summary'),original=document.createElement('p');
    summary.textContent='对照原文与逐段校对';original.textContent=`原文：\n${segment.original}`;details.append(summary,original);
    const candidate=document.createElement('p');candidate.textContent=`模型候选：\n${segment.candidate}`;details.append(candidate);
    if(segment.body_differences?.length){
      const note=document.createElement('p');note.className='difference-note';note.textContent='正文差异（位置按去除标点和空白后的原文计数）：';
      const list=document.createElement('ul');list.className='differences';
      for(const change of segment.body_differences){const row=document.createElement('li');
        if(change.operation==='insert')row.textContent=`${change.start?'第 '+change.start+' 字后':'开头'}多出：“${change.candidate}”`;
        else if(change.operation==='delete')row.textContent=`漏掉第 ${change.start+1}–${change.end} 字：“${change.source}”`;
        else row.textContent=`第 ${change.start+1}–${change.end} 字：“${change.source}”被改为“${change.candidate}”`;
        list.append(row);}
      details.append(note,list);
    }
    const edit=document.createElement('textarea');edit.className='review-editor';edit.rows=4;
    edit.setAttribute('aria-label',`第 ${segment.index+1} 段校对文本`);edit.value=segment.candidate||segment.text;
    const actions=document.createElement('div');actions.className='review-actions';
    const save=document.createElement('button');save.className='quiet';save.type='button';save.textContent='保存校对';
    const feedback=document.createElement('span');feedback.className='review-feedback';
    save.addEventListener('click',async()=>{
      if(!lastResult)return;
      const normalized=value=>value.normalize('NFC').replace(/[\p{P}\s—…·]/gu,'');
      const proposed=edit.value;
      if(!proposed.trim()){feedback.textContent='校对文本不能为空。';return;}
      if(data.task==='punctuate'&&normalized(proposed)!==normalized(segment.original)){
        feedback.textContent='正文仍有改字或漏字，请修正后再保存。';return;
      }
      save.disabled=true;edit.disabled=true;
      if(data.document_id){try{const response=await fetch(`/api/documents/${data.document_id}/segments/${segment.index}`,{method:'PATCH',headers:auth(),body:JSON.stringify({text:proposed,revision:segment.revision||0})});const updated=await response.json();if(!response.ok)throw Error(updated.detail||'保存失败');segment.revision=updated.revision;}catch(error){feedback.textContent=error.message;save.disabled=false;edit.disabled=false;return;}}
      edit.disabled=false;if(lastResult!==data)return;
      segment.text=proposed;segment.status='user_confirmed';text.textContent=segment.text;
      article.classList.remove('review');label.textContent=`第 ${segment.index+1} 段 · 已保存你的校对`;
      lastResult.text=assembledText(lastResult);
      lastResult.review_required=lastResult.segments.filter(row=>row.status==='review_required').length;
      lastResult.completed_segments=lastResult.segments.filter(row=>row.finish_reason!=='length'||row.status==='user_confirmed').length;
      feedback.textContent=data.document_id?'已保存到本机，将用于导出。':'已保存，将用于导出。';save.disabled=true;
      $('status').textContent=lastResult.review_required?`已保存 · ${lastResult.review_required} 段仍需复核`:'已保存校对';
      refreshHistory();
    });
    edit.addEventListener('input',()=>{save.disabled=false;feedback.textContent='';});
    actions.append(save,feedback);details.append(edit,actions);article.append(details);$('result').append(article);
  }
  $('export').disabled=false;
}
function invalidateResult(message,keepDocument=false){if(!keepDocument){viewEpoch++;rememberDocument(null);}lastResult=null;$('export').disabled=true;$('result').replaceChildren();$('result').classList.add('empty');$('result').textContent='处理结果会显示在这里。';$('status').textContent=message;}
async function refreshProgress(id){
  if(activeRequestId!==id||stopping)return;
  try{const response=await fetch('/api/progress/'+encodeURIComponent(id),{headers:auth()});if(!response.ok)return;
    const data=await response.json();if(activeRequestId!==id||stopping)return;
    $('status').textContent=`已完成 ${data.completed_segments}/${data.total_segments} 段 · 用时 ${Math.floor(data.elapsed_seconds)} 秒`;
  }catch{/* Progress is optional; the main request owns errors and results. */}
}
$('source').addEventListener('input',()=>{$('count').textContent=`${$('source').value.length} / 3000`;invalidateResult('原文已变更，请重新处理。');});
$('task').addEventListener('change',()=>invalidateResult('处理方式已变更，请重新处理。'));
$('history').addEventListener('change',async()=>{if(controller||!$('history').value)return;try{const restored=await loadDocument($('history').value,true);if(!restored)return;$('status').textContent=`已恢复 ${restored.completed_segments}/${restored.total_segments} 段 · 点击开始可继续`;}catch(error){$('status').textContent=error.message;}});
$('access-form').addEventListener('submit',async event=>{event.preventDefault();token=$('token').value.trim();$('access-error').textContent='';try{await connect();sessionStorage.setItem('classical_token',token);}catch(error){$('access-error').textContent=error.message;}});
$('lock').addEventListener('click',()=>{if(controller)return;token='';sessionStorage.removeItem('classical_token');$('token').value='';$('access').showModal();});
$('run').addEventListener('click',async()=>{
  if(controller)return;viewEpoch++;const text=$('source').value;
  if(!text.trim()){$('status').textContent='请先输入原文。';return;}
  if(!token){$('access').showModal();return;}
  controller=new AbortController();activeRequestId=crypto.randomUUID();cancelFeedback='已停止，原文保留。';stopping=false;
  $('run').disabled=true;$('cancel').disabled=false;$('source').disabled=true;$('task').disabled=true;$('file').disabled=true;$('export').disabled=true;
  $('history').disabled=true;
  invalidateResult('正在逐段处理，请稍候……',true);
  const progressId=activeRequestId;progressTimer=setInterval(()=>refreshProgress(progressId),1000);
  try{
    const prepared=await fetch('/api/documents',{method:'POST',headers:auth(),body:JSON.stringify({task:$('task').value,text}),signal:controller.signal});const metadata=await prepared.json();if(!prepared.ok)throw Error(metadata.detail||'无法保存本机文档。');rememberDocument(metadata.document_id);
    const response=await fetch('/api/process',{method:'POST',headers:auth(),body:JSON.stringify({task:$('task').value,text,request_id:activeRequestId}),signal:controller.signal});
    const data=await response.json();if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:`处理失败（${response.status}）`);
    output(data);$('status').textContent=data.complete===false?'部分输出未完成 · 点击开始可继续':data.review_required?`完成 · ${data.review_required} 段需复核`:'完成 · 正文一致';
  }catch(error){let recovered=null;if(documentId){try{recovered=await loadDocument(documentId);}catch{}}$('status').textContent=(error.name==='AbortError'?cancelFeedback:error.message)+(recovered?` · 已保存 ${recovered.completed_segments}/${recovered.total_segments} 段`:'');}
  finally{clearInterval(progressTimer);progressTimer=null;controller=null;activeRequestId=null;stopping=false;$('run').disabled=false;$('cancel').disabled=true;$('source').disabled=false;$('task').disabled=false;$('file').disabled=false;$('history').disabled=false;await refreshHistory();}
});
$('cancel').addEventListener('click',async()=>{
  if(!controller)return;const current=controller,id=activeRequestId,stopController=new AbortController();
  stopping=true;
  const timer=setTimeout(()=>stopController.abort(),5000);$('cancel').disabled=true;$('status').textContent='正在停止处理……';
  try{
    const response=await fetch('/api/cancel',{method:'POST',headers:auth(),body:JSON.stringify({request_id:id}),signal:stopController.signal});
    if(!response.ok)throw Error('取消请求未确认');
  }catch{if(controller===current)cancelFeedback='已停止等待，服务端取消未确认，请稍后重试。';}
  finally{clearTimeout(timer);current.abort();}
});
$('file').addEventListener('change',async()=>{const file=$('file').files[0];if(!file)return;try{if(file.size>65536)throw Error('文件过大，请分段后导入。');let text;try{text=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer());}catch{throw Error('TXT 不是有效 UTF-8，请转换编码后重试。');}if(text.length>3000)throw Error('文件超过 3000 字符，请分段处理。');$('source').value=text;$('source').dispatchEvent(new Event('input'));}catch(error){$('status').textContent=error.message;}$('file').value='';});
$('export').addEventListener('click',()=>{if(!lastResult)return;const link=document.createElement('a'),url=URL.createObjectURL(new Blob([lastResult.text],{type:'text/plain;charset=utf-8'}));link.href=url;link.download='古文研读结果.txt';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
connect().catch(error=>{$('access-error').textContent=error.message;if(!$('access').open)$('access').showModal();});
