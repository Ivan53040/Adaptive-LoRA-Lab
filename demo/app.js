const $ = (id) => document.getElementById(id);
let state = null, activeWorld = "lora", running = false, reviews = [], elapsed = null;
let toolSchemas = [], lastResponses = [], connected = true;
let notice = {key:"initialNotice", values:{}, error:false};
let lastRouteResult=null, routeNotice={key:'routeInitial',values:{}};
const examples = {
  flash: "Please turn on the flashlight.",
  contact: "Create a contact named Ada Lovelace with email ada@example.com.",
  calendar: "Schedule a calendar event titled 'Demo Review' on October 9th, 2026 at 2 PM.",
  email: "Send an email to demo@example.com with subject 'LoRA Demo' and body 'Hello from my local adapter.'",
  multi: "Please open my Wi-Fi settings and turn off the flashlight."
};
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function pre(value) { return node("pre", typeof value === "string" ? value : JSON.stringify(value, null, 2)); }
function pct(value) { return typeof value === "number" ? (value * 100).toFixed(2) + "%" : "—"; }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}
function renderNotice() {
  $("notice").textContent = t(state?.status === "paused" && notice.key === "initialNotice" ? "pausedNotice" : notice.key,notice.values);
  $("notice").className = "notice" + (notice.error ? " error" : "");
}
function setNotice(key, values = {}, error = false) {
  notice = {key,values,error}; renderNotice();
}
function worldView() {
  if (!state) return;
  const world = state.worlds[activeWorld];
  $("flashlight").classList.toggle("on",world.flashlight_on);
  $("flashText").textContent = t(world.flashlight_on ? "on" : "off");
  $("wifiText").textContent = t(world.wifi_settings_open ? "open" : "closed");
  $("eventsCount").textContent = world.events.length;
  $("contactsCount").textContent = world.contacts.length;
  $("mailCount").textContent = world.outbox.length;
  const list = $("stateList"); list.replaceChildren();
  const items = [];
  world.events.forEach((x) => items.push([t("calendar")+" · "+x.title,x.datetime]));
  world.contacts.forEach((x) => items.push([t("contacts")+" · "+x.first_name+" "+x.last_name,x.email || x.phone_number || ""]));
  world.outbox.forEach((x) => items.push([t("mockEmail")+" · "+x.subject,x.to]));
  world.map_queries.forEach((x) => items.push([t("mapQuery"),x]));
  items.slice(-10).reverse().forEach(([title,detail]) => {
    const item = node("div",undefined,"state-item");
    item.append(node("strong",title),node("span",detail)); list.append(item);
  });
  if (!items.length) list.append(node("p",t("worldEmpty"),"empty"));
}
function renderStatus() {
  const status = !connected ? "error" : state?.status || "loading";
  $("status").className = "status "+status;
  $("status").lastElementChild.textContent = t(!connected ? "disconnected" : status === "ready" ? "ready" : status === "paused" ? "paused" : status === "error" ? "loadError" : "loading");
  $("run").disabled = running || !connected || status !== "ready";
  if (typeof renderCodingStatus === 'function') renderCodingStatus();
  if (typeof renderRouterStatus === 'function') renderRouterStatus();
  $("reviewTab").textContent = t("reviewTab",{count:state?.review_count ?? reviews.length});
  if (!state) return;
  const metrics = state.benchmark.lora || {};
  $("formatMetric").textContent = pct(metrics.valid_tool_call_rate);
  $("toolMetric").textContent = pct(metrics.tool_selection_accuracy);
  $("exactMetric").textContent = pct(metrics.full_call_exact_match_rate);
  const mock = state.mock_summary.lora;
  $("mockMetric").textContent = mock ? pct(mock.strict_mock_execution_success/mock.examples) : "—";
  $("modelLabel").textContent = !$('coding').classList.contains('hidden') ? state.model+' · NF4 · '+t('sharedAdapters') : state.model+" · NF4 · LoRA r"+state.adapter_rank+(state.training_rows ? " · "+t("trainedRows",{count:state.training_rows.toLocaleString()}) : "");
  $("modelLabel").title = state.adapter_path;
}
function updateStatus(data) {
  state = data; connected = true;
  if (data.status === "error") setNotice("requestError",{message:data.error},true);
  if (!$("referenceTime").value) $("referenceTime").value = data.now;
  renderStatus(); renderNotice(); worldView();
}
function showResponses(responses) {
  lastResponses = responses;
  $("resultCount").textContent = t("responseCount",{count:responses.length});
  const grid = node("div",undefined,"response-grid");
  responses.forEach((r) => {
    const box = node("article",undefined,"response");
    const header = node("div",undefined,"response-heading");
    header.append(node("span",r.variant === "lora" ? "TOOL-USE LORA" : "BASE"),
                  node("span",r.latency_seconds.toFixed(2)+"s · "+r.generated_tokens+" tokens","muted"));
    box.append(header);
    const badges = node("div",undefined,"badges");
    [[r.format_valid,"badgeFormat"],[r.native_schema_valid,"badgeSchema"],[r.execution.ok,"badgeMock"]].forEach(([ok,key]) => badges.append(node("span",t(key)+(ok ? " ✓" : " ✕"),"badge"+(ok ? "" : " bad"))));
    box.append(badges,pre(r.calls));
    r.execution.results.forEach((x) => {
      const result = node("div",undefined,"call-result");
      result.append(node("strong",x.name),pre(x.result)); box.append(result);
    });
    r.execution.errors.forEach((x) => box.append(node("div",x.message,"call-result error-detail")));
    const details = node("details"); details.append(node("summary",t("rawOutput")),pre(r.output_text)); box.append(details);
    grid.append(box);
  });
  $("responses").replaceChildren(grid);
}
function openTab(name, syncHash = true) {
  if (!["play","auto","coding","pipeline","review","tools"].includes(name)) return;
  document.querySelectorAll("[data-tab]").forEach((button) => button.classList.toggle("active",button.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("hidden",panel.id !== name));
  if (syncHash) history.replaceState(null,"",['pipeline','coding','auto'].includes(name) ? '#'+name : location.pathname+location.search);
  renderStatus();
}
function reviewView() {
  const filter = $("reviewFilter").value;
  const list = $("reviewList"); list.replaceChildren();
  const visible = reviews.filter((r) => filter === "all" || (filter === "runtime" ? r.mock_execution && !r.mock_execution.ok : r.category === filter));
  visible.forEach((r) => {
    const details = node("details",undefined,"review-item");
    const summary = node("summary");
    summary.append(node("span","#"+r.test_index,"review-index"),node("span",r.prompt,"review-prompt"));
    const failed = r.mock_execution && !r.mock_execution.ok;
    summary.append(node("span",t(failed ? "filterRuntime" : r.category === "invalid_format" ? "filterFormat" : "argumentDifference"),"badge"+(failed ? " bad" : ""))); details.append(summary);
    const pair = node("div",undefined,"pair");
    const gold = node("div"); gold.append(node("label",t("referenceCalls")),pre(r.expected_calls));
    const predicted = node("div"); predicted.append(node("label",t("predictedCalls")),pre(r.predicted_calls)); pair.append(gold,predicted);
    details.append(pair);
    if (r.differences.length) details.append(node("label",t("fieldDifferences"),"muted"),pre(r.differences));
    if (failed) details.append(pre(r.mock_execution.errors));
    if (r.output_text) {
      const raw = node("details"); raw.append(node("summary",t("savedOutput")),pre(r.output_text)); details.append(raw);
    }
    details.append(node("p",t("reviewNote"),"muted")); list.append(details);
  });
  if (!visible.length) list.append(node("p",t("reviewEmpty"),"empty"));
}
function toolsView() {
  $("toolsList").replaceChildren();
  toolSchemas.forEach(({function:fn}) => {
    const card = node("article",undefined,"tool-card");
    const description = language === "zh" ? t("tool_"+fn.name) : fn.description;
    card.append(node("h3",fn.name),node("p",description));
    const required = fn.parameters.required || [];
    const args = Object.keys(fn.parameters.properties).map((name) => name+(required.includes(name) ? " *" : " ("+t("optional")+")"));
    card.append(node("code",args.length ? args.join(" · ") : t("noArguments"))); $("toolsList").append(card);
  });
}
function setLanguage(value) {
  if (value !== "en" && value !== "zh") return;
  language = value;
  try { localStorage.setItem("adaptive-lora-language",language); } catch (_) {}
  localizeStatic(); renderStatus(); renderNotice(); worldView(); reviewView(); toolsView(); pipelineView();
  if (lastResponses.length) showResponses(lastResponses);
  if (typeof codingResultsView === 'function') { renderCodingStatus(); codingResultsView(); codingReviewView(); }
  if (typeof renderRouterResults === 'function') renderRouterResults();
}
document.querySelectorAll("[data-language]").forEach((button) => button.addEventListener("click",() => setLanguage(button.dataset.language)));
document.querySelectorAll("[data-example]").forEach((button) => button.addEventListener("click",() => { $("prompt").value = examples[button.dataset.example]; }));
document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click",() => openTab(button.dataset.tab)));
document.querySelectorAll("[data-world]").forEach((button) => button.addEventListener("click",() => {
  activeWorld = button.dataset.world;
  document.querySelectorAll("[data-world]").forEach((x) => x.classList.toggle("active",x === button)); worldView();
}));
$("run").addEventListener("click",async () => {
  running = true; renderStatus(); setNotice("generating"); const start = performance.now();
  elapsed = setInterval(() => { $("timer").textContent = ((performance.now()-start)/1000).toFixed(1)+"s"; },100);
  try {
    const result = await api("/api/run",{
      prompt:$("prompt").value,mode:document.querySelector("input[name=mode]:checked").value,
      reference_time:$("referenceTime").value,max_new_tokens:Number($("maxTokens").value)
    });
    updateStatus(result.state); showResponses(result.responses); setNotice("finished");
  } catch (error) { setNotice("requestError",{message:error.message},true); }
  finally { clearInterval(elapsed); running = false; renderStatus(); }
});
$("reset").addEventListener("click",async () => {
  try { updateStatus(await api("/api/reset",{})); setNotice("resetNotice"); }
  catch (error) { setNotice("requestError",{message:error.message},true); }
});
$("reviewFilter").addEventListener("change",reviewView);
async function bootstrap() {
  toolSchemas = await api("/api/tools"); toolsView();
  reviews = await api("/api/review"); reviewView();
  updateStatus(await api("/api/status"));
}
localizeStatic(); pipelineView(); renderStatus(); renderNotice();
if (location.hash === "#pipeline" || location.hash.startsWith("#stage-")) openTab("pipeline",false);
else if (location.hash === '#coding') openTab('coding',false);
else if (location.hash === '#auto') openTab('auto',false);
bootstrap().catch((error) => setNotice("requestError",{message:error.message},true));
setInterval(async () => {
  try { updateStatus(await api("/api/status")); }
  catch (_) { connected = false; renderStatus(); }
},2000);

function renderRouterStatus() {
  if (!$('routeRun')) return;
  $('routeRun').disabled=running || !connected || state?.status!=='ready';
  $('routePreview').disabled=running || !connected;
  $('routeVariant').querySelector('option[value=expanded]').disabled=!state?.coding?.expanded;
  $('routeNotice').textContent=t(state?.status==='paused' && routeNotice.key==='routeInitial' ? 'pausedNotice' : routeNotice.key,routeNotice.values);
  if (state && !$('routeTime').value) $('routeTime').value=state.now;
}
function renderRouterResults() {
  if (!lastRouteResult) return;
  const result=lastRouteResult, route=result.route, card=node('div');
  card.append(node('h3',t('routeDomain_'+route.domain)+' → '+(route.adapter==='tool' ? 'Tool-use LoRA' : route.adapter==='coding' ? t(route.coding_variant==='expanded' ? 'codingExpanded' : 'codingOriginal') : route.adapter==='base' ? 'Base' : t('routeNoModel'))),
              node('p',t(route.reason_key),'muted'),node('p',t('routeTiming',{route:route.routing_ms.toFixed(3),total:result.end_to_end_seconds?.toFixed(2) || '—'}),'muted'));
  if (route.matched_tools.length) card.append(pre(route.matched_tools));
  if (result.kind==='tools') {
    const response=result.responses[0];
    card.append(node('span',t(response.execution.ok ? 'codeChecksPass':'codeChecksFail'),'badge'+(response.execution.ok ? '' : ' bad')),pre(response.calls),pre(response.execution.results));
  } else if (result.kind==='python') {
    const response=result.responses[0];
    card.append(node('span',t('pythonSyntax')+' '+(response.syntax_valid ? '✓':'×'),'badge'+(response.syntax_valid ? '' : ' bad')),pre(response.program));
    if (response.execution) card.append(node('span',t(response.checks_status==='unavailable' ? 'codeChecksUnavailable' : response.execution.passed ? 'codeChecksPass':'codeChecksFail'),'badge'),pre(response.execution));
    else card.append(node('p',t('codeChecksNotRun'),'muted'));
  } else if (result.kind==='general') card.append(pre(result.response.answer));
  if (result.responses || result.response) {
    const raw=node('details');raw.append(node('summary',t('rawOutput')),pre(result.responses?.[0]?.output_text || result.response?.output_text || ''));card.append(raw);
  }
  $('routeOutput').replaceChildren(card);
}
async function runRouter(generate) {
  running=true;renderStatus();routeNotice={key:'routeWorking',values:{}};renderRouterStatus();
  try {
    lastRouteResult=await api('/api/route',{prompt:$('routePrompt').value,generate,
      allow_experimental_coding:$('routeExperimental').checked,run_tests:generate && $('routeChecks').checked,
      coding_variant:$('routeVariant').value,
      tests:$('routeAssertions').value,reference_time:$('routeTime').value});
    updateStatus(lastRouteResult.state);renderRouterResults();routeNotice={key:'routeFinished',values:{}};
  } catch(error) {routeNotice={key:'requestError',values:{message:error.message}};}
  finally {running=false;renderStatus();renderRouterStatus();}
}
$('routeRun').addEventListener('click',()=>runRouter(true));
$('routePreview').addEventListener('click',()=>runRouter(false));
document.querySelectorAll('[data-route-example]').forEach(button=>button.addEventListener('click',()=>{
  const values={tool:'Please open my Wi-Fi settings and turn off the flashlight.',
    python:'def square(x):\n    """Return x multiplied by itself."""\n',general:'What is 3 + 4? Answer with just the number.',
    clarify:'Turn on the flashlight and write a Python function.'};
  $('routePrompt').value=values[button.dataset.routeExample];
}));
renderRouterStatus();
