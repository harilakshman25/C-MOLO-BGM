"""Offline listening page; timestamps are entered by a human, never inferred."""
import hashlib
import json
from pathlib import Path

root=Path('outputs/forest-narration')
anchors=json.loads(Path('examples/forest_anchors.json').read_text())
segments=json.loads((root/'segments.json').read_text())
data=dict(narration_sha256=hashlib.sha256((root/'narration.wav').read_bytes()).hexdigest(),
          cues=[dict(id=cid,word=a['word'],**segments[a['segment']]) for cid,a in anchors.items()])
page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Mark six sound cues</title>
<style>body{font:18px system-ui;max-width:950px;margin:40px auto;padding:0 20px;color:#183340;background:#f5f8fa}h1{font-size:32px}article{padding:18px 0;border-bottom:1px solid #ccd6dc}button,input,select{font:inherit;padding:8px;margin:5px}button{cursor:pointer}audio{width:100%}label{display:block}textarea{width:100%;height:170px;font:14px monospace}.note{color:#456}strong{color:#087f8c}</style>
<h1>Mark six sound cues</h1><p>Play each sentence. Pause just after the highlighted trigger word finishes, then click <b>Use paused time</b>. You can replay and adjust the seconds. These are your listening annotations.</p>
<audio id="audio" controls preload="auto" src="narration.wav"></audio>
<label>Playback speed <select id="speed"><option value="1">Normal</option><option value="0.75">Slower (0.75x)</option><option value="0.5">Slow (0.5x)</option></select></label>
<div id="cues"></div><p id="message" role="status"></p><button id="export">Prepare timing notes</button><button id="download" disabled>Download JSON</button>
<p class="note">Copy the prepared JSON below into the chat, or attach the downloaded file. No upload or server connection is used. Do not close the page before saving your notes.</p><textarea id="result" readonly aria-label="Prepared timing notes"></textarea>
<script>
const data=__DATA__, audio=document.getElementById('audio'), inputs={};let stopAt=null;
document.getElementById('speed').onchange=e=>audio.playbackRate=Number(e.target.value);
audio.addEventListener('timeupdate',()=>{if(stopAt!==null&&audio.currentTime>=stopAt){audio.pause();stopAt=null;}});
for(const cue of data.cues){
 const row=document.createElement('article');const title=document.createElement('h2');title.textContent='Trigger: '+cue.word;row.append(title);
 const description=document.createElement('p');description.textContent=cue.text;row.append(description);
 const range=document.createElement('p');range.className='note';range.textContent=`Sentence: ${cue.start_s.toFixed(2)}–${cue.end_s.toFixed(2)} seconds`;row.append(range);
 const play=document.createElement('button');play.textContent='Play sentence';play.onclick=async()=>{audio.pause();audio.currentTime=cue.start_s;stopAt=cue.end_s+.2;try{await audio.play();}catch(e){document.getElementById('message').textContent='Use the audio player to start playback.'}};row.append(play);
 const field=document.createElement('input');field.type='number';field.step='0.01';field.min=cue.start_s;field.max=cue.end_s;field.setAttribute('aria-label',cue.word+' end time in seconds');inputs[cue.id]=field;
 const mark=document.createElement('button');mark.textContent='Use paused time';mark.onclick=()=>{audio.pause();stopAt=null;field.value=audio.currentTime.toFixed(3);};row.append(mark,field);
 document.getElementById('cues').append(row);
}
document.getElementById('export').onclick=()=>{
 const marks={};for(const cue of data.cues){const value=inputs[cue.id].value;const n=Number(value);if(value===''||!Number.isFinite(n)||n<cue.start_s||n>cue.end_s){document.getElementById('message').textContent='Enter a time within the sentence for: '+cue.word;document.getElementById('download').disabled=true;return;}marks[cue.id]=n;}
 document.getElementById('result').value=JSON.stringify({method:'human-listening-word-end',narration_sha256:data.narration_sha256,word_end_s:marks},null,2);
 document.getElementById('message').textContent='All six times are ready. Copy below or download.';document.getElementById('download').disabled=false;
};
document.getElementById('download').onclick=()=>{const url=URL.createObjectURL(new Blob([document.getElementById('result').value],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='forest-manual-times.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></html>'''
path=root/'manual-timing.html'
path.write_text(page.replace('__DATA__',json.dumps(data).replace('<','\\u003c')),encoding='utf-8')
print(path.resolve())
