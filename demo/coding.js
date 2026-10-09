let codeResponses=[], codeReviews=[], codeNotice={key:'codeInitial',values:{}};
function renderCodingStatus() {
  const choice=$('codeVariant').value;
  const ready=connected && state?.status==='ready' && (document.querySelector('input[name=codeMode]:checked').value==='base' || (choice==='expanded' ? state?.coding?.expanded?.ready : state?.coding?.ready));
  $('codeVariant').querySelector('option[value=expanded]').disabled=!state?.coding?.expanded;
  $('codeRun').disabled=running || !ready;
  $('codeNotice').textContent=t(codeNotice.key,codeNotice.values);
  if (state?.coding?.error) $('codeNotice').textContent=t('requestError',{message:state.coding.error});
  const table=$('codeMetrics'); table.replaceChildren();
  ['base','tool','coding'].forEach((variant) => {
    const metric=state?.coding?.metrics?.[variant], row=node('tr');
    row.append(node('td',variant==='coding' ? t('codingExperimental') : variant==='tool' ? 'Tool-use LoRA' : 'Base'));
    row.append(node('td',metric ? metric.passed+' / '+metric.examples+' · '+pct(metric.pass_at_1) : '—'));
    row.append(node('td',metric ? pct(metric.syntax_valid_rate) : '—'));
    table.append(row);
  });
  const expandedMetric=PIPELINE_FACTS.finalHumanEval?.find(row=>row.condition==='expanded');
  if (expandedMetric) {
    const row=node('tr');row.append(node('td',t('codingExpanded')),
      node('td',expandedMetric.passed+' / '+expandedMetric.examples+' · '+pct(expandedMetric.pass_at_1)),
      node('td',pct(expandedMetric.syntax_valid_rate)));table.append(row);
  }
}
function codingResultsView() {
  if (!codeResponses.length) return;
  const grid=node('div',undefined,'response-grid');
  codeResponses.forEach((record) => {
    const card=node('article',undefined,'response-card');
    card.append(node('h3',record.variant==='coding' ? t(record.coding_variant==='expanded' ? 'codingExpanded' : 'codingOriginal') : 'Base'),
      node('p',record.latency_seconds.toFixed(2)+'s · '+record.generated_tokens+' tokens · '+t('adapterSwitch')+' '+record.adapter_switch_ms.toFixed(2)+'ms','muted'));
    const badges=node('div',undefined,'badges');
    badges.append(node('span',t('pythonSyntax')+' '+(record.syntax_valid ? '✓' : '×'),'badge'+(record.syntax_valid ? '' : ' bad')));
    if (record.execution) badges.append(node('span',t(record.checks_status==='unavailable' ? 'codeChecksUnavailable' : record.execution.passed ? 'codeChecksPass' : 'codeChecksFail'),'badge'+(record.checks_status==='unavailable' ? ' progress-badge' : record.execution.passed ? '' : ' bad')));
    else badges.append(node('span',t('codeChecksNotRun'),'badge'));
    card.append(badges,pre(record.program));
    if (record.syntax_error) card.append(pre(record.syntax_error));
    if (record.execution && !record.execution.passed) card.append(pre(record.execution));
    const raw=node('details'); raw.append(node('summary',t('rawOutput')),pre(record.output_text)); card.append(raw);
    grid.append(card);
  });
  $('codeResponses').replaceChildren(grid);
}
function codingReviewView() {
  const expanded=new Set(Array.from($('codeReviewList').querySelectorAll('details[id]')).filter(item=>item.open).map(item=>item.id));
  const filter=$('codeReviewFilter').value;
  const visible=codeReviews.filter((row) => filter==='failures' ? !row.coding_passed :
    filter==='gained' ? row.transition==='coding_only_passed' :
    filter==='lost' ? row.transition==='base_only_passed' : filter==='all' || row.category===filter);
  $('codeReviewCount').textContent=t('codeCaseCount',{count:visible.length});
  const list=$('codeReviewList'); list.replaceChildren();
  visible.forEach((row) => {
    const details=node('details',undefined,'review-item'), summary=node('summary');
    details.id='coding-case-'+row.test_index; details.open=expanded.has(details.id);
    summary.append(node('span',row.task_id,'review-index'),node('span',t('codeReviewStatus',{base:row.base_passed ? '✓':'×',coding:row.coding_passed ? '✓':'×'}),'review-prompt'));
    summary.append(node('span',t('codeCategory_'+row.category),'badge'+(row.coding_passed ? '' : ' bad'))); details.append(summary);
    details.append(node('label',t('codeFunctionPrefix')),pre(row.prompt));
    const pair=node('div',undefined,'pair'), base=node('div'), coding=node('div');
    base.append(node('label','Base'),pre(row.base_completion));
    coding.append(node('label',t('codingExperimental')),pre(row.coding_completion)); pair.append(base,coding); details.append(pair);
    if (row.coding_error_type) details.append(pre({type:row.coding_error_type,message:row.coding_error}));
    const reference=node('details'); reference.append(node('summary',t('codeReference')),pre(row.reference_completion)); details.append(reference);
    details.append(node('p',t('codeReviewNote'),'muted')); list.append(details);
  });
  if (!visible.length) list.append(node('p',t('reviewEmpty'),'empty'));
}
$('codeRun').addEventListener('click',async () => {
  running=true; renderStatus(); codeNotice={key:'codeGenerating',values:{}}; renderCodingStatus();
  const start=performance.now(), timer=setInterval(() => {$('codeTimer').textContent=((performance.now()-start)/1000).toFixed(1)+'s';},100);
  try {
    const result=await api('/api/code',{prefix:$('codePrefix').value,
      mode:document.querySelector('input[name=codeMode]:checked').value,
      coding_variant:$('codeVariant').value,
      max_new_tokens:Number($('codeTokens').value),run_tests:$('codeTestEnable').checked,tests:$('codeTests').value});
    updateStatus(result.state); codeResponses=result.responses; codingResultsView(); codeNotice={key:'codeFinished',values:{}};
  } catch (error) { codeNotice={key:'requestError',values:{message:error.message}}; }
  finally {clearInterval(timer); running=false; renderStatus(); renderCodingStatus();}
});
$('codeReviewFilter').addEventListener('change',codingReviewView);
$('codeVariant').addEventListener('change',renderCodingStatus);
document.querySelectorAll('input[name=codeMode]').forEach(input=>input.addEventListener('change',renderCodingStatus));
api('/api/coding-review').then((rows) => {codeReviews=rows; codingReviewView();}).catch((error) => {
  codeNotice={key:'requestError',values:{message:error.message}}; renderCodingStatus();
});
renderCodingStatus();
