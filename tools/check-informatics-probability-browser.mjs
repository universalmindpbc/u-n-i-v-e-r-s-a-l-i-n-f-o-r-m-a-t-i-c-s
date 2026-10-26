import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from '/Users/benkorbel/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';

const base = 'https://staging.universalmind.tv/informatics/';
const localCandidate = process.env.INFORMATICS_LOCAL_CANDIDATE === '1';
const candidateRoot = path.resolve(fileURLToPath(new URL('../Informatics/',import.meta.url)));
const output = process.env.INFORMATICS_QA_OUTPUT || '/private/tmp/m9-m18-probability-staging';
await fs.mkdir(output, {recursive:true});
const server = await chromium.launchServer({
  executablePath:'/Users/benkorbel/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell',
  headless:true, args:['--mute-audio'],
});
assert.ok(server.process().spawnargs.includes('--mute-audio'));
const browser = await chromium.connect(server.wsEndpoint());
const results = [];
try {
  for (const modulus of [9,18]) for (const viewport of [{width:1440,height:1000}, {width:390,height:844}]) {
    const context = await browser.newContext({viewport, reducedMotion:'reduce'});
    if(localCandidate) await context.route(base+'**',async route=>{
      const relative=decodeURIComponent(new URL(route.request().url()).pathname.slice('/informatics/'.length))||'index.html';
      const file=path.resolve(candidateRoot,relative);
      if(!file.startsWith(candidateRoot+'/')) return route.abort();
      const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.py':'text/plain'};
      try {await route.fulfill({status:200,body:await fs.readFile(file),contentType:types[path.extname(file)]||'application/octet-stream'});}
      catch {await route.continue();}
    });
    const download=async url=>{
      if(!localCandidate)return context.request.get(url);
      const file=path.resolve(candidateRoot,new URL(url).pathname.slice('/informatics/'.length));
      const body=await fs.readFile(file,'utf8');
      return {status:()=>200,json:async()=>JSON.parse(body),text:async()=>body};
    };
    // Preserve the app's fullscreen-container flow without native fullscreen.
    await context.addInitScript(() => {
      let fullscreen = null;
      Object.defineProperty(document, 'fullscreenElement', {get:()=>fullscreen});
      Element.prototype.requestFullscreen = async function() {fullscreen=this;document.dispatchEvent(new Event('fullscreenchange'));};
      document.exitFullscreen = async () => {fullscreen=null;document.dispatchEvent(new Event('fullscreenchange'));};
    });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
      const expected = JSON.parse(await fs.readFile(new URL(`../Informatics/pilot-v1/m${modulus}-probability.json`, import.meta.url),'utf8'));
      const format = value => String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
      const url = base + `?atlas=1&record=fibonacci-m${modulus}-p24#geometry-summary`;
      await page.goto(url, {waitUntil:'domcontentloaded'});
      await page.waitForFunction(()=>!document.querySelector('#launch').disabled);
      await page.locator('#launch').click();
      const frame = page.frameLocator('#atlas-frame');
      const section = frame.locator('#atlas-probability');
      await section.waitFor({state:'visible'});
      assert.equal(await section.locator('h3').innerText(), 'Probability');
      assert.equal(await section.locator('.probability-value').innerText(), expected.display);
      assert.ok((await section.innerText()).includes(expected.method.beta_values.join(', ')));
      assert.equal(await section.locator('blockquote').innerText(), '“We are exploring equilateral triangles, honeycomb theorem, and carbon structures in relationship to this Atlas for useful directions for computation in OXTR research.”');
      assert.equal(await section.evaluate(e=>e.nextElementSibling.id), 'm9-comparison');
      await page.waitForFunction(()=>{const v=document.querySelector('#film');return !v.paused && v.currentTime > 0.2;}, null, {timeout:45000});
      await page.evaluate(()=>{
        window.__probabilityFilm=document.querySelector('#film');
        window.__probabilityMediaEvents=[];
        for(const type of ['pause','emptied','loadstart']) __probabilityFilm.addEventListener(type,()=>__probabilityMediaEvents.push(type));
        window.__probabilityFullscreen=document.fullscreenElement;
      });
      const nativeRecord=JSON.parse(await fs.readFile(new URL(`../Informatics/pilot-v1/m${modulus}.json`,import.meta.url),'utf8'));
      const observation=frame.locator('#summary-beta-observation');
      assert.equal(await observation.isVisible(),true);
      assert.equal(await observation.locator('p').innerText(),nativeRecord.native_beta_observation.text);
      await observation.screenshot({path:path.join(output,`m${modulus}-observation-summary-${viewport.width}.png`)});
      await frame.getByRole('tab',{name:'Beta',exact:true}).click();
      const betaNote=frame.locator('#beta-observation');
      assert.equal(await betaNote.isVisible(),true);
      assert.equal(await betaNote.locator('p').innerText(),nativeRecord.native_beta_observation.text);
      assert.equal(await frame.locator('#sequence').innerText(),nativeRecord.beta.display_cycle.join(' → '));
      assert.equal(await betaNote.evaluate(e=>e.scrollWidth<=e.clientWidth+1),true);
      await betaNote.screenshot({path:path.join(output,`m${modulus}-observation-beta-${viewport.width}.png`)});
      await frame.getByRole('tab',{name:'Alpha',exact:true}).click();
      assert.equal(await betaNote.isVisible(),false);
      await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
      await section.screenshot({path:path.join(output, `m${modulus}-probability-${viewport.width}.png`)});
      await section.getByRole('button', {name:'View calculation'}).click();
      const dialog = frame.locator('.probability-reader');
      assert.equal(await dialog.evaluate(e=>e.open), true);
      assert.equal(await dialog.getByRole('heading', {name:'Probability',exact:true}).count(), 1);
      assert.equal(await dialog.locator('ol li').count(), 4);
      assert.ok((await dialog.innerText()).includes(format(expected.counts.all_arrangements)));
      assert.ok((await dialog.innerText()).includes(format(expected.counts.matching_arrangements)));
      assert.ok((await dialog.innerText()).includes(expected.calculation.formula));
      assert.equal(page.url(), url);
      assert.equal(await dialog.evaluate(e=>{
        const r=e.getBoundingClientRect();return r.left>=0 && r.right<=innerWidth+1 && r.top>=0 && r.bottom<=innerHeight+1 && e.scrollWidth<=e.clientWidth+1;
      }), true);
      await dialog.screenshot({path:path.join(output, `m${modulus}-calculation-${viewport.width}.png`)});
      await dialog.getByText('Verification', {exact:true}).click();
      assert.match(await dialog.innerText(), /Three exact counting methods agree/);
      const jsonURL = await dialog.getByRole('link', {name:'Calculation JSON'}).getAttribute('href');
      const jsonResponse = await download(new URL(jsonURL,base).href);
      assert.equal(jsonResponse.status(),200);
      const probability = await jsonResponse.json();
      assert.equal(probability.result.fraction, expected.result.fraction);
      assert.equal(typeof probability.counts.all_arrangements,'string');
      assert.deepEqual(probability.method.beta_values,expected.method.beta_values);
      const codeURL = await dialog.getByRole('link', {name:'Python calculation'}).getAttribute('href');
      const codeResponse = await download(new URL(codeURL,base).href);
      assert.equal(codeResponse.status(),200);
      assert.match(await codeResponse.text(), /labelled_partition_count/);
      await dialog.getByRole('button', {name:'Close probability calculation'}).click();
      assert.equal(await dialog.evaluate(e=>e.open),false);
      assert.equal(await section.locator('button').evaluate(e=>e===document.activeElement),true);
      await section.locator('button').click();
      await page.keyboard.press('Escape');
      assert.equal(await dialog.evaluate(e=>e.open),false);
      assert.equal(await section.locator('button').evaluate(e=>e===document.activeElement),true);

      for (const [modulus,period] of [[27,72],[36,24],[45,120],[1,1],[48,24]]) {
        await frame.locator('#research-record').selectOption(`fibonacci-m${modulus}-p${period}`);
        await frame.locator('.intro h1').filter({hasText:new RegExp(`^M${modulus} / P${period}$`)}).waitFor();
        assert.equal(await section.isVisible(),false);
        assert.equal(await section.innerText(),'');
        assert.equal(await frame.locator('#summary-beta-observation').isVisible(),false);
        assert.equal(await frame.locator('#beta-observation p').innerText(),'');
        if(modulus%9!==0){await frame.getByRole('tab',{name:'Beta',exact:true}).click();assert.equal(await frame.locator('#beta-display').isVisible(),false);await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();}
      }
      await frame.locator('#research-record').selectOption('fibonacci-m9-p24');
      await section.waitFor({state:'visible'});
      await frame.getByRole('tab',{name:'Beta',exact:true}).click();
      assert.equal(await frame.locator('#sequence').innerText(),'4 → 8 → 7 → 5 → 1 → 2');
      assert.equal(await section.isVisible(),false);
      await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
      await section.waitFor({state:'visible'});
      await frame.locator('#atlas-science-summary [data-science-register]').click();
      assert.equal(await frame.locator('.science-register tbody tr').count(),14);
      await page.keyboard.press('Escape');
      const countRows = [[9,24,4,1,true],[18,24,4,1,true],[27,72,12,3,false],[36,24,4,1,false],[45,120,20,5,false]];
      for (const [m,p,total,primary,doubling] of countRows) {
        await frame.locator('#research-record').selectOption(`fibonacci-m${m}-p${p}`);
        await frame.locator('.intro h1').filter({hasText:new RegExp(`^M${m} / P${p}$`)}).waitFor();
        await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();
        const overview = frame.locator('#other-geometry-summary');
        await overview.waitFor({state:'visible'});
        assert.deepEqual(await overview.locator('dd').allTextContents(),[String(total),String(primary),doubling?'Present':'Absent']);
        assert.equal(await section.isVisible(),m===9||m===18);
        const expectedRecord=JSON.parse(await fs.readFile(new URL(`../Informatics/pilot-v1/m${m}.json`,import.meta.url),'utf8'));
        const projection=expectedRecord.beta_m9_view;
        await frame.getByRole('tab',{name:'Beta',exact:true}).click();
        const switcher=frame.locator('#beta-display');
        assert.equal(await switcher.isVisible(),true);
        assert.equal(await switcher.getByRole('button',{name:`Native M${m}`,exact:true}).getAttribute('aria-pressed'),'true');
        const nativeSrc=await frame.locator('#diagram').getAttribute('src');
        assert.equal(await frame.locator('#sequence').innerText(),projection.native_values.join(' → '));
        await switcher.getByRole('button',{name:'Reduced to M9',exact:true}).click();
        assert.equal(await frame.locator('#sequence').innerText(),projection.reduced_values.join(' → '));
        assert.ok((await frame.locator('#description').innerText()).includes('111 multiples of nine'));
        assert.ok((await frame.locator('#diagram').getAttribute('src')).includes(`/m${m}/beta-m9.svg`));
        assert.ok((await frame.locator('#svg-download').getAttribute('href')).includes('beta-m9.svg'));
        assert.ok((await frame.locator('#png-download').getAttribute('href')).includes('beta-m9.png'));
        if(!await frame.locator('#calculation').isVisible())await frame.getByText('Inspect the calculation',{exact:true}).click();
        assert.ok((await frame.locator('#calculation').innerText()).includes(projection.native_values.join(' → ')));
        await frame.getByText('Inspect the calculation',{exact:true}).click();
        if(projection.repeat_count>1)assert.ok((await frame.locator('#convention').innerText()).includes(`${projection.repeat_count} repeated cycles`));
        assert.equal(await switcher.getByRole('button',{name:'Reduced to M9',exact:true}).getAttribute('aria-pressed'),'true');
        assert.equal(await switcher.evaluate(e=>e.scrollWidth<=e.clientWidth+1),true);
        assert.equal(await frame.locator('body').evaluate(e=>e.scrollWidth<=innerWidth+1),true);
        await frame.locator('.workspace').screenshot({path:path.join(output,`m${m}-beta-reduced-${viewport.width}.png`)});
        await frame.getByRole('tab',{name:'Alpha',exact:true}).click();
        assert.equal(await switcher.isVisible(),false);
        await frame.getByRole('tab',{name:'Beta',exact:true}).click();
        assert.equal(await switcher.getByRole('button',{name:'Reduced to M9',exact:true}).getAttribute('aria-pressed'),'true');
        await switcher.getByRole('button',{name:`Native M${m}`,exact:true}).click();
        assert.equal(await frame.locator('#diagram').getAttribute('src'),nativeSrc);
        assert.equal(await frame.locator('#sequence').innerText(),projection.native_values.join(' → '));
        if(m===18){
          await frame.getByRole('tab',{name:'Epsilon',exact:true}).click();
          assert.equal(await switcher.isVisible(),false);
          await frame.locator('#epsilon-reduced').click();
          assert.ok((await frame.locator('#diagram').getAttribute('src')).includes('epsilon-m9.svg'));
          await frame.getByRole('tab',{name:'Beta',exact:true}).click();
          assert.equal(await switcher.getByRole('button',{name:`Native M${m}`,exact:true}).getAttribute('aria-pressed'),'true');
        }
        for(const kind of ['alpha','gamma','epsilon']){
          const view=expectedRecord.m9_sequence_views[kind];
          await frame.getByRole('tab',{name:kind[0].toUpperCase()+kind.slice(1),exact:true}).click();
          assert.equal(await frame.locator('#'+kind+'-display').isVisible(),true);
          await frame.locator('#'+kind+'-native').click();
          const nativeImage=await frame.locator('#diagram').getAttribute('src');
          if(kind==='epsilon'&&!expectedRecord.epsilon&&!expectedRecord.epsilon_path){
            assert.equal(await frame.locator('#candidate-controls').isVisible(),true);
            assert.match(await frame.locator('#description').innerText(),/No qualifying native Epsilon path/);
          }
          await frame.locator('#'+kind+'-reduced').click();
          assert.equal(await frame.locator('#sequence').innerText(),view.reduced_values.join(' → '));
          assert.equal(await frame.locator('#candidate-controls').isVisible(),false);
          assert.ok((await frame.locator('#diagram').getAttribute('src')).includes('/'+kind+'-m9.svg'));
          assert.ok((await frame.locator('#svg-download').getAttribute('href')).includes('/'+kind+'-m9.svg'));
          assert.ok((await frame.locator('#png-download').getAttribute('href')).includes('/'+kind+'-m9.png'));
          assert.equal(await frame.locator('#png-download').isVisible(),true);
          assert.equal(await frame.locator('#'+kind+'-reduced').getAttribute('aria-pressed'),'true');
          if(view.repeat_count>1)assert.match(await frame.locator('#convention').innerText(),new RegExp(view.repeat_count+' repeated cycles'));
          if(kind==='epsilon')assert.match(await frame.locator('#description').innerText(),m===9?/M9 is the reference/:/each reduced 24-position cycle/);
          assert.equal(await frame.locator('body').evaluate(e=>e.scrollWidth<=innerWidth+1),true);
          await frame.locator('.workspace').screenshot({path:path.join(output,`m${m}-${kind}-reduced-${viewport.width}.png`)});
          await frame.getByRole('tab',{name:'Beta',exact:true}).click();
          assert.equal(await frame.locator('#beta-native').getAttribute('aria-pressed'),'true');
          await frame.getByRole('tab',{name:kind[0].toUpperCase()+kind.slice(1),exact:true}).click();
          assert.equal(await frame.locator('#'+kind+'-reduced').getAttribute('aria-pressed'),'true');
          await frame.locator('#'+kind+'-native').click();
          if(!(kind==='epsilon'&&!expectedRecord.epsilon&&!expectedRecord.epsilon_path))assert.equal(await frame.locator('#diagram').getAttribute('src'),nativeImage);
        }
        await frame.getByRole('tab',{name:'Geometry summary',exact:true}).click();

        await overview.screenshot({path:path.join(output,`m${m}-counts-${viewport.width}.png`)});
        await overview.getByRole('link',{name:'Inspect formations'}).click();
        await frame.locator('#candidate-picker').waitFor({state:'visible'});
        const entries=await frame.locator('#candidate-picker option').allTextContents();
        assert.equal(entries.filter(t=>t.startsWith('2D stellated octahedron')).length,total);
        assert.ok((await frame.locator('#description').innerText()).startsWith(total+' 2D stellated octahedrons'));
      }
      await frame.locator('#research-record').selectOption('m-summary');
      await frame.locator('#family-cards[aria-busy="false"]').waitFor();
      assert.equal(await frame.locator('.family-card').count(),5);
      for(const [i,[m,p,total,primary,doubling]] of countRows.entries()) {
        const card=frame.locator('.family-card').nth(i);
        assert.equal(await card.locator('.family-formation-counts').innerText(),`Native 2D stellated octahedrons: ${total} across Alpha · ${primary} in primary Gamma`);
        assert.equal(await card.locator('.family-beta-doubling').innerText(),`Native Beta doubling: ${doubling?'Present':'Absent'}`);
      }
      await frame.locator('#family-cards').screenshot({path:path.join(output,`family-counts-${viewport.width}.png`)});
      await frame.getByRole('button',{name:'Explore M18 P24',exact:true}).click();
      await frame.locator('.intro h1').filter({hasText:'M18 / P24'}).waitFor();
      const media = await page.evaluate(()=>({
        same:document.querySelector('#film')===__probabilityFilm,
        playing:!__probabilityFilm.paused,
        events:__probabilityMediaEvents,
        fullscreenPreserved:document.fullscreenElement===__probabilityFullscreen,
      }));
      assert.deepEqual(media,{same:true,playing:true,events:[],fullscreenPreserved:true});
      assert.deepEqual(errors,[]);
      results.push({modulus,viewport,probability:true,lightbox:true,downloads:true,originalAlignment:true,
        otherRecordsUnassigned:true,nativeBetaObservations:true,betaNativeReducedToggle:true,betaFullSequenceAndRepeats:true,betaExports:true,allFourNativeReducedViews:true,cycleBasedEpsilon:true,epsilonIndependent:true,formationCounts:true,familySummary:true,focusReturn:true,musicContinuous:true,nativeFullscreenInvoked:false,errors});
    } finally {
      await page.evaluate(()=>document.querySelectorAll('video,audio').forEach(e=>e.pause())).catch(()=>{});
      await context.close();
    }
  }
} finally {
  await browser.close();
  await server.close();
}
await fs.writeFile(path.join(output,'results.json'),JSON.stringify({passed:true,mode:localCandidate?'local_candidate_interception':'deployed_staging',results},null,2));
console.log(JSON.stringify({passed:true,mode:localCandidate?'local_candidate_interception':'deployed_staging',results},null,2));
