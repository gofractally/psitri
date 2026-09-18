const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('/Users/dlarimer/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const dir = __dirname;
  const browser = await chromium.launch({headless:true, executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page = await browser.newPage({viewport:{width:1280,height:1000}});
  const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8767/comparison.html');
  if (await page.locator('#engines input').count() !== 7) throw Error('Missing engine controls');
  const metrics = await page.locator('#metric option').evaluateAll(os=>os.map(o=>o.value));
  for (const m of metrics) {
    await page.selectOption('#metric',m);
    if (await page.locator('.barrow').count() !== 42) throw Error('Missing bars: '+m);
    if (!await page.locator('#units').textContent()) throw Error('Missing units');
  }
  await page.selectOption('#metric','write');
  const boxes = page.locator('#engines input');
  for (let i=0;i<7;i++) await boxes.nth(i).uncheck();
  if (!await page.locator('#chart').textContent().then(x=>x.includes('Select an engine'))) throw Error('No empty-selection state');
  await boxes.nth(5).check(); await boxes.nth(6).check();
  if (await page.locator('.barrow').count() !== 12) throw Error('Small Engine selection failed');
  for (let i=0;i<7;i++) await boxes.nth(i).check();
  await page.screenshot({path:path.join(dir,'comparison.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  const overflow = await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
  if (overflow) throw Error('Mobile horizontal overflow');
  await page.screenshot({path:path.join(dir,'comparison-mobile.png'),fullPage:true});
  if (errors.length) throw Error(errors.join('\n'));
  const result = {status:'passed',engines:7,batch_sizes:6,bars:42,metrics:metrics.length,checkbox_filtering:true,mobile_width:390,horizontal_overflow:false,browser_errors:errors};
  fs.writeFileSync(path.join(dir,'chart-validation.json'),JSON.stringify(result,null,2)+'\n');
  await browser.close(); console.log(result);
})().catch(e=>{console.error(e);process.exit(1)});
