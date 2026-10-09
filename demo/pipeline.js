const PIPELINE_FACTS = {
  "selectedRank": 8,
  "rows": {
    "train": 1000,
    "validation": 100,
    "test": 961
  },
  "ranks": [
    {
      "rank": 8,
      "exact": 0.9,
      "format": 1,
      "mock": 1,
      "adapterMiB": 88.07552337646484
    },
    {
      "rank": 16,
      "exact": 0.89,
      "format": 1,
      "mock": 1,
      "adapterMiB": 176.08824920654297
    },
    {
      "rank": 32,
      "exact": 0.89,
      "format": 1,
      "mock": 1,
      "adapterMiB": 352.1119613647461
    }
  ],
  "counts": {
    "exact": 892,
    "format": 959,
    "tools": 956,
    "mock": 958
  },
  "seed": 42,
  "currentCounts": {
    "valid_tool_call": 961,
    "tools_correct": 960,
    "full_call_exact": 893,
    "mock_execution_ok": 961,
    "hit_token_limit": 0
  },
  "dataSizes": [
    {
      "name": "1k",
      "rows": 1000,
      "updates": 1000,
      "exact": 0.9,
      "format": 1.0,
      "mock": 1.0
    },
    {
      "name": "3k",
      "rows": 3000,
      "updates": 3000,
      "exact": 0.93,
      "format": 1.0,
      "mock": 1.0
    },
    {
      "name": "full",
      "rows": 8258,
      "updates": 8258,
      "exact": 0.92,
      "format": 1.0,
      "mock": 1.0
    }
  ],
  "codingResults": [
    {
      "variant": "base",
      "validationPassed": 17,
      "testPassed": 98,
      "testRate": 0.5975609756097561,
      "syntaxRate": 0.9451219512195121
    },
    {
      "variant": "tool",
      "validationPassed": 8,
      "testPassed": 89,
      "testRate": 0.5426829268292683,
      "syntaxRate": 0.8841463414634146
    },
    {
      "variant": "coding",
      "validationPassed": 19,
      "testPassed": 97,
      "testRate": 0.5914634146341463,
      "syntaxRate": 0.9634146341463414
    }
  ],
  "codingFollowup": [
    {
      "seed": 42,
      "scope": "all",
      "updates": 340,
      "passed": 19
    },
    {
      "seed": 43,
      "scope": "all",
      "updates": 340,
      "passed": 19
    },
    {
      "seed": 44,
      "scope": "all",
      "updates": 340,
      "passed": 19
    },
    {
      "seed": 42,
      "scope": "attention",
      "updates": 340,
      "passed": 17
    }
  ],
  "codingExpansion": [
    {
      "condition": "base",
      "validationPassed": 17,
      "testPassed": 224,
      "testRate": 0.4817204301075269,
      "syntaxValid": 406,
      "medianLatency": 2.01500000001397
    },
    {
      "condition": "pilot",
      "validationPassed": 19,
      "testPassed": 241,
      "testRate": 0.5182795698924731,
      "syntaxValid": 460,
      "medianLatency": 3.8589999999967404
    },
    {
      "condition": "expanded",
      "validationPassed": 17,
      "testPassed": 249,
      "testRate": 0.535483870967742,
      "syntaxValid": 461,
      "medianLatency": 3.1089999999967404
    }
  ],
  "codingAuditCount": 1587,
  "finalHumanEval": [
    {
      "condition": "base",
      "variant": "base",
      "split": "test",
      "examples": 164,
      "rows_sha256": "2759c4482a91f8102017f6944a0ec1ba59f866242d0ca46536590fc1a134702f",
      "passed": 98,
      "pass_at_1": 0.5975609756097561,
      "syntax_valid_rate": 0.9451219512195121,
      "token_limit_hit_rate": 0.03048780487804878,
      "median_latency_seconds": 2.2659999999959837,
      "peak_allocated_vram_gib": 5.8614630699157715,
      "generation": "one greedy candidate per frozen task; raw function completion,512-token budget, stop at next top-level definition/comment",
      "benchmark_scope": "untouched HumanEval164"
    },
    {
      "condition": "pilot",
      "variant": "coding",
      "split": "test",
      "examples": 164,
      "rows_sha256": "2759c4482a91f8102017f6944a0ec1ba59f866242d0ca46536590fc1a134702f",
      "passed": 97,
      "pass_at_1": 0.5914634146341463,
      "syntax_valid_rate": 0.9634146341463414,
      "token_limit_hit_rate": 0.018292682926829267,
      "median_latency_seconds": 3.6555000000007567,
      "peak_allocated_vram_gib": 5.94735860824585,
      "generation": "one greedy candidate per frozen task; raw function completion,512-token budget, stop at next top-level definition/comment",
      "benchmark_scope": "untouched HumanEval164"
    },
    {
      "condition": "expanded",
      "variant": "coding",
      "split": "test",
      "examples": 164,
      "rows_sha256": "2759c4482a91f8102017f6944a0ec1ba59f866242d0ca46536590fc1a134702f",
      "passed": 111,
      "pass_at_1": 0.676829268292683,
      "syntax_valid_rate": 0.9939024390243902,
      "token_limit_hit_rate": 0.0,
      "median_latency_seconds": 3.507500000006985,
      "peak_allocated_vram_gib": 5.94735860824585,
      "generation": "one greedy candidate per frozen task; raw function completion,512-token budget, stop at next top-level definition/comment",
      "benchmark_scope": "untouched HumanEval164"
    }
  ],
  "finalCodingSeeds": [
    {
      "seed": 42,
      "updates": 3000,
      "validation_passed": 17,
      "test_passed": 249,
      "test_rate": 0.535483870967742
    },
    {
      "seed": 43,
      "updates": 3000,
      "validation_passed": 21,
      "test_passed": 252,
      "test_rate": 0.5419354838709678
    },
    {
      "seed": 44,
      "updates": 3000,
      "validation_passed": 17,
      "test_passed": 247,
      "test_rate": 0.5311827956989247
    }
  ],
  "finalToolConditions": [
    {
      "seed": 42,
      "rank": 8,
      "updates": 1000,
      "exact": 90,
      "mock": 100,
      "peak_training_gib": 11.801278114318848
    },
    {
      "seed": 42,
      "rank": 16,
      "updates": 1000,
      "exact": 89,
      "mock": 100,
      "peak_training_gib": 12.146652698516846
    },
    {
      "seed": 42,
      "rank": 32,
      "updates": 1000,
      "exact": 89,
      "mock": 100,
      "peak_training_gib": 12.864206314086914
    },
    {
      "seed": 43,
      "rank": 8,
      "updates": 1000,
      "exact": 88,
      "mock": 100,
      "peak_training_gib": 11.800731658935547
    },
    {
      "seed": 43,
      "rank": 16,
      "updates": 1000,
      "exact": 86,
      "mock": 99,
      "peak_training_gib": 12.147522449493408
    },
    {
      "seed": 43,
      "rank": 32,
      "updates": 1000,
      "exact": 82,
      "mock": 100,
      "peak_training_gib": 12.859785556793213
    },
    {
      "seed": 44,
      "rank": 8,
      "updates": 1000,
      "exact": 90,
      "mock": 100,
      "peak_training_gib": 11.80165958404541
    },
    {
      "seed": 44,
      "rank": 16,
      "updates": 1000,
      "exact": 90,
      "mock": 100,
      "peak_training_gib": 12.14606237411499
    },
    {
      "seed": 44,
      "rank": 32,
      "updates": 1000,
      "exact": 79,
      "mock": 97,
      "peak_training_gib": 12.855110168457031
    }
  ],
  "finalToolRanks": {
    "8": {
      "mean": 89.33333333333333,
      "minimum": 88,
      "maximum": 90,
      "conditions": 3
    },
    "16": {
      "mean": 88.33333333333333,
      "minimum": 86,
      "maximum": 90,
      "conditions": 3
    },
    "32": {
      "mean": 83.33333333333333,
      "minimum": 79,
      "maximum": 89,
      "conditions": 3
    }
  }
};
const PIPELINE_STAGES = [
  {id:"data", n:1, facts:["Qwen3-8B-Base","RTX 5090 Laptop · 24 GB","Seed 42"], file:"data/raw/mobile-actions/dataset.jsonl"},
  {id:"train", n:2, file:"training/train_tool_lora.py"},
  {id:"repair", n:3, facts:["@completionLoss","@trainableFp32","lm_head + transformer LoRA"], file:"training/repair_protocol.py"},
  {id:"ranks", n:4, facts:["@epochsUpdates","@batchTwo","α / r = 2 · dropout 0.05","Head LR 1e-3 · body LR 1e-4"], file:"experiments/fair_rank.py"},
  {id:"test", n:5, file:"evaluation/rank_validation.py · evaluation/audit_protocol.py"},
  {id:"audit", n:6},
  {id:"demo", n:7, file:".\\.venv\\Scripts\\python.exe demo\\server.py"},
  {id:"sizes", n:8, file:"experiments/data_size.py · evaluation/audit_protocol.py"},
  {id:"coding", n:9, facts:["@codingCounts","r8 · α16 · NF4 / FP32 LoRA"], file:"experiments/coding_study.py · evaluation/review_coding.py"},
  {id:"switching", n:10, file:"demo/server.py · runtime/coding_engine.py"},
  {id:"routing", n:11, file:"runtime/router.py · evaluation/router_benchmark.py"},
  {id:"research", n:12, file:"experiments/coding_followup.py · experiments/coding_expansion.py"},
  {id:"expanded-live", n:13, file:"demo/expanded_server.py · evaluation/router_expanded_benchmark.py"},
  {id:"remaining-gpu", n:14, file:"experiments/gpu_remaining.py · configs/research_remaining/"}
];
function pipelineView() {
  const map = $("pipelineMap"); map.replaceChildren();
  [["data","mapData"],["train","mapTrain"],["repair","mapRepair"],["ranks","mapRank"],["test","mapTest"],["demo","mapDemo"],["sizes","mapDataSize"],["coding","mapCoding"]].forEach(([id,key], index) => {
    if (index) map.append(node("span","→"));
    const link = node("a",t(key)); link.href = "#stage-" + id; map.append(link);
  });
  const stages = $("pipelineStages");
  const expanded = new Set(Array.from(stages.querySelectorAll("article")).filter((article) => article.querySelector("details")?.open).map((article) => article.id));
  stages.replaceChildren();
  const split = (items) => {
    const grid = node("div",undefined,"split-grid");
    items.forEach(([value,label]) => {
      const box = node("div"); box.append(node("strong",value),node("span",t(label))); grid.append(box);
    });
    return grid;
  };
  PIPELINE_STAGES.forEach((stage) => {
    const article = node("article",undefined,"pipeline-stage" + (stage.planned ? " planned" : ""));
    article.id = "stage-" + stage.id;
    article.append(node("div",String(stage.n).padStart(2,"0"),"stage-number"));
    const card = node("div",undefined,"card"), heading = node("div",undefined,"section-title");
    heading.append(node("h2",t("p"+stage.n+"Title")),node("span",t(stage.inProgress ? "inProgress" : stage.planned ? "planned" : "done"),"badge" + (stage.inProgress ? " progress-badge" : stage.planned ? " planned-badge" : "")));
    card.append(heading,node("p",t("p"+stage.n+"Body")));
    if (stage.facts) {
      const tags = node("div",undefined,"pipeline-facts");
      stage.facts.forEach((fact) => tags.append(node("span",fact.startsWith("@") ? t(fact.slice(1)) : fact))); card.append(tags);
    }
    if (stage.id === "data") card.append(split([[PIPELINE_FACTS.rows.train.toLocaleString(),"trainingRows"],[PIPELINE_FACTS.rows.validation,"validationRows"],[PIPELINE_FACTS.rows.test,"testRows"]]));
    if (stage.id === "ranks") {
      const wrap = node("div",undefined,"table-scroll"), table = node("table",undefined,"pipeline-table");
      table.append(node("caption",t("rankTableCaption")));
      const head = node("thead"), hr = node("tr");
      ["Rank",t("tableExact"),t("tableFormat"),t("tableMock"),t("tableSize")].forEach((label) => hr.append(node("th",label)));
      head.append(hr); table.append(head); const body = node("tbody");
      PIPELINE_FACTS.ranks.forEach((rank) => {
        const row = node("tr",undefined,rank.rank === PIPELINE_FACTS.selectedRank ? "selected" : "");
        const label = node("td","r"+rank.rank);
        if (rank.rank === PIPELINE_FACTS.selectedRank) label.append(node("span",t("selected"),"badge"));
        row.append(label);
        [pct(rank.exact),pct(rank.format),pct(rank.mock),rank.adapterMiB.toFixed(1)+" MiB"].forEach((value) => row.append(node("td",value)));
        body.append(row);
      });
      table.append(body); wrap.append(table); card.append(wrap);
    }
    if (stage.id === "test") {
      card.append(split([[PIPELINE_FACTS.counts.exact+" / "+PIPELINE_FACTS.rows.test,"tableExact"],[PIPELINE_FACTS.counts.format+" / "+PIPELINE_FACTS.rows.test,"tableFormat"],[PIPELINE_FACTS.counts.mock+" / "+PIPELINE_FACTS.rows.test,"tableMock"]]),node("p",t("testNote"),"pipeline-note"));
    }
    if (stage.id === "audit" || stage.id === "research" || stage.id === "routing" || stage.id === "sizes" || stage.id === "coding" || stage.id === "switching") card.append(node("p",t("p"+stage.n+"Note"),"pipeline-note"));
    if (stage.id === "coding") {
      const wrap=node('div',undefined,'table-scroll'), table=node('table',undefined,'pipeline-table');
      table.append(node('caption',t('codeComparisonCaption')));
      const head=node('thead'),hr=node('tr');
      ['codeMode','codeValidation','codeHeldOut','pythonSyntax'].forEach((key)=>hr.append(node('th',t(key))));
      head.append(hr); table.append(head); const body=node('tbody');
      PIPELINE_FACTS.codingResults.forEach((result)=>{
        const row=node('tr'); row.append(node('td',result.variant==='coding' ? t('codingExperimental') : result.variant==='tool' ? 'Tool-use LoRA':'Base'));
        [result.validationPassed+' / 32',result.testPassed+' / 164 · '+pct(result.testRate),pct(result.syntaxRate)].forEach(value=>row.append(node('td',value)));
        body.append(row);
      });
      table.append(body);wrap.append(table);card.append(wrap);
    }
    if (stage.id === 'research') {
      const followWrap=node('div',undefined,'table-scroll'),followTable=node('table',undefined,'pipeline-table');
      followTable.append(node('caption',t('researchFollowupCaption')));
      const followHead=node('thead'),followHeader=node('tr');
      ['researchTargets','researchSeed','tableUpdates','codeValidation'].forEach(key=>followHeader.append(node('th',t(key))));
      followHead.append(followHeader);followTable.append(followHead);const followBody=node('tbody');
      PIPELINE_FACTS.codingFollowup.forEach(result=>{
        const row=node('tr');[t(result.scope==='all' ? 'researchAllTargets' : 'researchAttentionTargets'),String(result.seed),String(result.updates),result.passed+' / 32'].forEach(value=>row.append(node('td',value)));followBody.append(row);
      });
      followTable.append(followBody);followWrap.append(followTable);card.append(followWrap);
      const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'pipeline-table');
      table.append(node('caption',t('researchExpansionCaption')));
      const head=node('thead'),hr=node('tr');
      ['codeMode','codeValidation','researchMbppTest','pythonSyntax','researchLatency'].forEach(key=>hr.append(node('th',t(key))));
      head.append(hr);table.append(head);const body=node('tbody');
      PIPELINE_FACTS.codingExpansion.forEach(result=>{
        const row=node('tr');[result.condition==='base' ? 'Base' : t(result.condition==='pilot' ? 'researchPilot' : 'researchExpanded'),result.validationPassed+' / 32',result.testPassed+' / 465 · '+pct(result.testRate),result.syntaxValid+' / 465',result.medianLatency.toFixed(3)+' s'].forEach(value=>row.append(node('td',value)));body.append(row);
      });
      table.append(body);wrap.append(table);card.append(wrap,node('p',t('researchAudit',{count:PIPELINE_FACTS.codingAuditCount.toLocaleString()}),'pipeline-note'));
    }
    if (stage.id === 'coding' || stage.id === 'switching') {
      const link=node('button',t('openCoding'),'text-button pipeline-link');link.type='button';
      link.addEventListener('click',()=>openTab('coding'));card.append(link);
    }
    if (stage.id === 'remaining-gpu') {
      const finalTable=(caption,headings,rows)=>{
        const wrap=node('div',undefined,'table-scroll'),table=node('table',undefined,'pipeline-table');
        table.append(node('caption',t(caption)));const head=node('thead'),hr=node('tr');
        headings.forEach(key=>hr.append(node('th',t(key))));head.append(hr);table.append(head);
        const body=node('tbody');rows.forEach(values=>{const row=node('tr');values.forEach(value=>row.append(node('td',value)));body.append(row);});
        table.append(body);wrap.append(table);card.append(wrap);
      };
      finalTable('finalHumanCaption',['codeMode','codeHeldOut','pythonSyntax'],PIPELINE_FACTS.finalHumanEval.map(row=>[
        row.condition==='base' ? 'Base' : t(row.condition==='pilot' ? 'codingOriginal' : 'codingExpanded'),
        row.passed+' / 164 · '+pct(row.pass_at_1),pct(row.syntax_valid_rate)]));
      finalTable('finalCodingCaption',['researchSeed','tableUpdates','codeValidation','researchMbppTest'],PIPELINE_FACTS.finalCodingSeeds.map(row=>[
        String(row.seed),String(row.updates),row.validation_passed+' / 32',row.test_passed+' / 465 · '+pct(row.test_rate)]));
      finalTable('finalToolCaption',['finalRank','finalSeed42','finalSeed43','finalSeed44','finalMean','finalRange'],[8,16,32].map(rank=>{
        const values=PIPELINE_FACTS.finalToolConditions.filter(row=>row.rank===rank),stats=PIPELINE_FACTS.finalToolRanks[String(rank)];
        return ['r'+rank,...values.map(row=>row.exact.toFixed(2)+'%'),stats.mean.toFixed(2)+'%',stats.minimum+'–'+stats.maximum+'%'];
      }));
      card.append(node('p',t('finalEvidence'),'pipeline-note'));
    }
    if (stage.id === "sizes") {
      const wrap = node("div",undefined,"table-scroll"), table = node("table",undefined,"pipeline-table");
      table.append(node("caption",t("dataSizeCaption")));
      const head = node("thead"), hr = node("tr");
      ["tableRows","tableUpdates","tableExact","tableFormat","tableMock"].forEach((key) => hr.append(node("th",t(key))));
      head.append(hr); table.append(head); const body = node("tbody");
      PIPELINE_FACTS.dataSizes.forEach((size) => {
        const row = node("tr",undefined,size.name === "3k" ? "selected" : "");
        const label = node("td",size.rows.toLocaleString());
        if (size.name === "3k") label.append(node("span",t("selected"),"badge"));
        row.append(label);
        [size.updates.toLocaleString(),pct(size.exact),pct(size.format),pct(size.mock)].forEach((value) => row.append(node("td",value)));
        body.append(row);
      });
      table.append(body); wrap.append(table); card.append(wrap);
      card.append(split([[PIPELINE_FACTS.currentCounts.full_call_exact+" / 961","tableExact"],[PIPELINE_FACTS.currentCounts.valid_tool_call+" / 961","tableFormat"],[PIPELINE_FACTS.currentCounts.mock_execution_ok+" / 961","tableMock"]]));
      const review = node("button",t("openReview"),"text-button pipeline-link");
      review.type = "button"; review.addEventListener("click",() => openTab("review")); card.append(review);
    }
    if (stage.id === "demo") {
      const flow = node("div",undefined,"runtime-flow");
      ["runtimePrompt","runtimeGeneration","runtimeParse","runtimeValidate","runtimeExecute"].forEach((key,index) => {
        if (index) flow.append(node("b","→")); flow.append(node("span",t(key)));
      });
      card.append(flow);
    }
    if (stage.file) {
      const details = node("details"); details.open = expanded.has(article.id); details.append(node("summary",t("p"+stage.n+"DetailTitle")),node("p",t("p"+stage.n+"Detail")),node("code",stage.file)); card.append(details);
    }
    article.append(card); stages.append(article);
  });
  const artifacts = $("pipelineArtifacts"); artifacts.replaceChildren();
  [
    ["artifactRecipe","configs/fair_rank/r8.json · r16.json · r32.json"],
    ["artifactCompare","outputs/fair-rank-completion-1k/comparison.md"],
    ["artifactPredictions","outputs/data-size-r8/3k/evaluation/test-961-tokens256-predictions.jsonl"],
    ["artifactAudit","outputs/data-size-r8/3k/evaluation/test-error-audit.json"],
    ["artifactRankAudit","outputs/fair-rank-completion-1k/r8/evaluation/test-error-audit.json"],
    ["artifactResume",".\\.venv\\Scripts\\python.exe experiments\\fair_rank.py"],
    ["artifactDataSize","outputs/data-size-r8/comparison.md"],
    ["artifactDataProgress",".\\.venv\\Scripts\\python.exe experiments\\data_size.py --status"],
    ["artifactCoding",".\\.venv\\Scripts\\python.exe -u experiments\\coding_study.py"],
    ["artifactCodingResults","outputs/coding-mbpp-r8-fp32/comparison.md"],
    ["artifactCodingReview","outputs/coding-mbpp-r8-fp32/evaluation/test-case-review.md"],
    ["artifactRouter","outputs/router-v1/comparison.md"],
    ["artifactFollowup",".\\.venv\\Scripts\\python.exe experiments\\coding_followup.py --status"],
    ["artifactExpansion",".\\.venv\\Scripts\\python.exe experiments\\coding_expansion.py --status"],
    ["artifactExpansionReview","outputs/coding-expansion-3k/test-case-review.md"],
    ["artifactExpansionAudit","outputs/coding-expansion-3k/independent-audit.json"],
    ["artifactExpandedLive","outputs/router-expanded-v1/comparison.md"],
    ["artifactRemainingGpu",".\\.venv\\Scripts\\python.exe experiments\\gpu_remaining.py --status"],
    ["artifactFinalSummary","outputs/gpu-remaining-20261007/final-summary.md"]
  ].forEach(([label,path]) => {
    const item = node("div"); item.append(node("span",t(label)),node("code",path)); artifacts.append(item);
  });
}
