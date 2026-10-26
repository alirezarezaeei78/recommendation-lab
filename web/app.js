'use strict';
const $=id=>document.getElementById(id);let saved=null;let requestVersion=0;
const ids=value=>value.split(',').map(x=>x.trim()).filter(Boolean);
for(const id of ['liked','seen','method','count']){$(id).addEventListener('change',()=>{requestVersion++;saved=null;$('json').disabled=true;$('csv').disabled=true;$('rows').replaceChildren();$('summary').textContent='Inputs changed. Generate a new ranking before exporting.';});}
fetch('/api/catalog').then(r=>r.json()).then(d=>{
  $('condition').textContent=(d.synthetic?'Authored synthetic demo':'Your local input data')+' · local-only · BPR seed '+(d.bpr_training?.seed??'unavailable');
  $('catalog').textContent='Available IDs: '+d.items.join(', ');
  for(const user of d.users||[]){const option=document.createElement('option');option.value=user;option.textContent=user;$('profile').append(option);}
}).catch(()=>{$('condition').textContent='Could not load the local catalog. Check that app.py is running.';});
$('profile').addEventListener('change',()=>{requestVersion++;const selected=Boolean($('profile').value);for(const id of ['liked','seen']){$(id).disabled=selected;$(id).value='';} saved=null;$('json').disabled=true;$('csv').disabled=true;$('rows').replaceChildren();$('summary').textContent=selected?'Generate to inspect this training user’s past history and ranking.':'Enter a custom history; BPR requires a fitted training-user profile.';});
$('run').addEventListener('click',async()=>{
  const version=++requestVersion;
  saved=null;$('json').disabled=true;$('csv').disabled=true;$('rows').replaceChildren();$('summary').className='';
  try{
    const profile=$('profile').value;
    const request={method:$('method').value,n:Number($('count').value)};
    if(profile)request.user_id=profile;else{request.liked=ids($('liked').value);request.seen=ids($('seen').value);}
    const response=await fetch('/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
    const result=await response.json();if(version!==requestVersion)return;if(!response.ok)throw new Error(result.error);
    saved=result;if(profile){$('liked').value=result.liked.join(',');$('seen').value=result.seen.filter(x=>!result.liked.includes(x)).join(',');}
    $('summary').textContent=`${result.effective_method}${result.history_positive_count!==undefined?' · '+result.history_positive_count+' unique likes · threshold '+result.policy_min_positives:''} · ${result.candidate_count} candidates · ${result.ranking.length} returned${result.fallback?' · '+(result.fallback_reason||'Explicit popularity fallback: no positive neighbor score.'):''}`;
    for(const r of result.ranking){const row=document.createElement('tr');for(const text of [r.rank,r.item,r.score.toFixed(6),r.positive_users,r.contributors.map(x=>x[0]+': '+x[1].toFixed(6)).join('; ')||'—']){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}$('rows').append(row);}
    $('json').disabled=false;$('csv').disabled=false;
  }catch(error){if(version===requestVersion){$('summary').textContent=error.message;$('summary').className='error';}}
});
function download(kind){if(!saved)return;const data=kind==='csv'?saved.csv:JSON.stringify(saved,null,2);const url=URL.createObjectURL(new Blob([data],{type:kind==='csv'?'text/csv':'application/json'}));const a=document.createElement('a');a.href=url;a.download='recommendations.'+kind;a.click();URL.revokeObjectURL(url);}
$('json').addEventListener('click',()=>download('json'));$('csv').addEventListener('click',()=>download('csv'));
