import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {gunzipSync} from 'node:zlib';

function parseCsv(text) {
  const rows=[];let row=[],cell='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
    else if(c===','&&!quoted){row.push(cell);cell='';}
    else if(c==='\n'&&!quoted){row.push(cell.replace(/\r$/,''));rows.push(row);row=[];cell='';}
    else cell+=c;
  }
  return rows;
}

export async function checkTfbsExplorer({page,dialog,root,out,viewport}) {
  const dataRoot=path.join(root,'Informatics/pilot-v1/oxtr-tfbs-v1');
  const read=async f=>JSON.parse(await fs.readFile(path.join(dataRoot,f),'utf8'));
  const encode=await read('encode-clusters.json'),provenance=await read('encode-provenance.json');
  const predictions=JSON.parse(gunzipSync(await fs.readFile(path.join(dataRoot,'jaspar-predictions.json.gz'))));
  const mappings=await read('sequence-mapping.json'),brain=await read('study-additions-v1/braintf-peaks.json');
  const explorer=dialog.locator('.tfbs-explorer'),status=explorer.locator('.tfbs-data-status');
  const search=explorer.locator('#tfbs-data-search'),position=explorer.locator('#tfbs-data-position');
  const expectCount=async (count,unit)=>{
    await status.filter({hasText:new RegExp(`^${count.toLocaleString('en-AU')} ${unit} match`)}).waitFor({timeout:60000});
  };
  const choose=async (id,count,unit)=>{await explorer.locator(`[data-tfbs-dataset="${id}"]`).click();await expectCount(count,unit);};
  const clear=async()=>explorer.locator('[data-tfbs-data-clear]').click();
  const exportAndCheck=async(expected,id)=>{
    const download=page.waitForEvent('download');await explorer.locator('[data-tfbs-data-export="json"]').click();
    const saved=path.join(out,`${id}-data-${viewport.width}.json`);await(await download).saveAs(saved);
    const result=JSON.parse(await fs.readFile(saved,'utf8'));
    assert.equal(result.collection,id);assert.equal(result.count,expected.length);
    assert.deepEqual(result.records.map(r=>r.id).sort(),expected.map(r=>r.id).sort());
    assert.equal(result.reference.assembly,'GRCh38');
    assert.ok(result.records.every(r=>r.binding_evidence.status==={published:r.study_id==='braintf'?'regional_association_detected':'binding_detected',encode:'regional_association_detected',jaspar:'prediction_only',oreganno:'not_classified'}[id]));
    const csvDownload=page.waitForEvent('download');await explorer.locator('[data-tfbs-data-export="csv"]').click();
    const csvPath=path.join(out,`${id}-data-${viewport.width}.csv`);await(await csvDownload).saveAs(csvPath);
    const [header,...records]=parseCsv((await fs.readFile(csvPath,'utf8')).replace(/^\uFEFF/,''));
    assert.equal(records.length,expected.length);
    assert.deepEqual(records.map(r=>r[header.indexOf('id')]).sort(),expected.map(r=>r.id).sort());
    for(const r of records){assert.equal(r[header.indexOf('species')],'Homo sapiens');assert.equal(r[header.indexOf('coordinate_convention')],'0-based, end-exclusive');}
    if(id==='encode')assert.ok(records.every(r=>JSON.parse(r[header.indexOf('experiments')]).length>0));
    return result;
  };

  await explorer.getByRole('button',{name:'Retry collection',exact:true}).click();
  await expectCount(255,'regions');
  assert.equal(await explorer.locator('tbody tr').count(),50);
  await explorer.locator('[data-tfbs-data-page="next"]').click();
  assert.equal(await explorer.locator('.tfbs-data-pages span').textContent(),'Page 2 of 6');
  await search.fill('CTCF');const ctcf=encode.filter(r=>r.target==='CTCF');await expectCount(ctcf.length,'regions');
  const result=await exportAndCheck(ctcf,'encode');
  assert.ok(result.records.every(r=>r.experiments.length===provenance.filter(p=>p.cluster_id===r.id).length));
  await search.fill('PC-3');const pc3Ids=new Set(provenance.filter(p=>p.cell_type==='PC-3').map(p=>p.cluster_id));await expectCount(pc3Ids.size,'regions');
  await clear();await position.fill('chr3:8,770,000-8,771,000');
  await expectCount(encode.filter(r=>r.start_0based<8771000&&r.end_exclusive>8770000).length,'regions');
  await position.fill('chr3:8771000-8770000');await status.filter({hasText:'The end must be after the start'}).waitFor();
  assert.equal(await explorer.locator('[data-tfbs-data-export="json"]').isDisabled(),true);
  await clear();await position.fill('chr4:8770000');await expectCount(0,'regions');await clear();

  assert.equal(await explorer.locator('[data-tfbs-dataset]').count(),3);
  await choose('published',128,'observations');
  const published=await exportAndCheck([...mappings.map(r=>({id:r.record_id})),...brain],'published');
  assert.equal(published.publications.length,3);
  await search.fill(mappings[0].sequence);await expectCount(1,'observations');await clear();
  await search.fill('REST');await expectCount(brain.filter(r=>r.factor==='REST').length,'observations');await exportAndCheck(brain.filter(r=>r.factor==='REST'),'published');await clear();
  await position.fill('chr3:8769128-8771628');await expectCount(116,'observations');await clear();
  await search.fill('YBX1 OL_NEUN');await expectCount(0,'observations');await clear();
  await choose('jaspar',9164,'matches');
  assert.equal(await explorer.locator('tbody tr').count(),50);
  // Export all matching observations, not just the first visible page.
  if(viewport.width===1440)await exportAndCheck(predictions.filter(r=>r.score>=400),'jaspar');
  await explorer.locator('#tfbs-data-score').selectOption('0');await expectCount(217857,'matches');
  const below=predictions.find(r=>r.score<400);
  await search.fill(below.id);await expectCount(1,'matches');
  await exportAndCheck([below],'jaspar');
  await explorer.locator('#tfbs-data-score').selectOption('400');await expectCount(0,'matches');
  await clear();await expectCount(9164,'matches');
  await search.fill('not-a-real-factor');await expectCount(0,'matches');
  assert.equal(await explorer.locator('[data-tfbs-data-export="csv"]').isDisabled(),true);
  await clear();
  // The score control belongs only to JASPAR; changing collections resets pagination.
  await choose('encode',255,'regions');
  assert.equal(await explorer.locator('.tfbs-score-filter').isVisible(),false);
  await dialog.evaluate(e=>e.scrollTop=0);
  assert.equal(await dialog.evaluate(e=>e.scrollWidth<=e.clientWidth+1),true);
  await page.screenshot({path:path.join(out,`data-explorer-${viewport.width}.png`)});
}
