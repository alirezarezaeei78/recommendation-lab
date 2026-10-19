'use strict';
const $=id=>document.getElementById(id);let saved=null;
const ids=value=>value.split(',').map(x=>x.trim()).filter(Boolean);
fetch('/api/catalog').then(r=>r.json()).then(d=>{
  $('condition').textContent=d.synthetic?'Authored synthetic demo · local-only · 20 neighbors · shrinkage 10':'Your local input data · local-only · no outbound API';
  $('catalog').textContent='Available IDs: '+d.items.join(', ');
}).catch(()=>{$('condition').textContent='Could not load the local catalog. Check that app.py is running.';});
$('run').addEventListener('click',async()=>{
  saved=null;$('json').disabled=true;$('csv').disabled=true;$('rows').replaceChildren();$('summary').className='';
  try{
    const response=await fetch('/api/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({liked:ids($('liked').value),seen:ids($('seen').value),method:$('method').value,n:Number($('count').value)})});
    const result=await response.json();if(!response.ok)throw new Error(result.error);
    saved=result;$('summary').textContent=`${result.effective_method} · ${result.candidate_count} candidates · ${result.ranking.length} returned${result.fallback?' · Explicit popularity fallback: no positive neighbor score.':''}`;
    for(const r of result.ranking){const row=document.createElement('tr');for(const text of [r.rank,r.item,r.score.toFixed(6),r.positive_users,r.contributors.map(x=>x[0]+': '+x[1].toFixed(6)).join('; ')||'—']){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}$('rows').append(row);}
    $('json').disabled=false;$('csv').disabled=false;
  }catch(error){$('summary').textContent=error.message;$('summary').className='error';}
});
function download(kind){if(!saved)return;const data=kind==='csv'?saved.csv:JSON.stringify(saved,null,2);const url=URL.createObjectURL(new Blob([data],{type:kind==='csv'?'text/csv':'application/json'}));const a=document.createElement('a');a.href=url;a.download='recommendations.'+kind;a.click();URL.revokeObjectURL(url);}
$('json').addEventListener('click',()=>download('json'));$('csv').addEventListener('click',()=>download('csv'));
