// Adapt labels only in the portable viewer; preserve the frozen research UI files.
Object.assign(UI_TEXT.en, {
  paused: 'Reports only',
  pausedNotice: 'Browse completed results and route previews. Model generation is not included in this public snapshot.',
  initialNotice: 'Browse completed results and route previews. No model is loaded.',
  codeInitial: 'Saved Coding comparisons are available. Generation requires the separate local research environment.',
  footer: 'SAVED RESEARCH RESULTS / CPU ROUTE PREVIEWS',
  publicSnapshot: 'Completed research snapshot · no model downloads required',
  watchVideo: 'Watch the 3:28 project film ↗'
});
Object.assign(UI_TEXT.zh, {
  paused: '成果瀏覽模式',
  pausedNotice: '可瀏覽已完成的成果和路由預覽。此公開版本不載入模型或產生新回覆。',
  initialNotice: '可瀏覽已完成的成果和路由預覽，無需載入模型。',
  codeInitial: '可查看 Coding 的已儲存比較結果。模型生成需另設本機研究環境。',
  footer: '已完成研究成果 / CPU 路由預覽',
  publicSnapshot: '已完成的研究成果快照 · 無需下載模型',
  watchVideo: '觀看 3 分 28 秒介紹影片 ↗'
});
if (!location.hash) location.hash = 'pipeline';
const reportBanner = document.createElement('div');
reportBanner.className = 'card';
reportBanner.style.cssText = 'max-width:1200px;margin:20px auto;padding:16px 24px;display:flex;gap:20px;flex-wrap:wrap;justify-content:space-between';
const reportLabel = document.createElement('span');
reportLabel.dataset.i18n = 'publicSnapshot';
reportLabel.textContent = t('publicSnapshot');
const filmLink = document.createElement('a');
filmLink.href = '/media/adaptive-lora-lab-star-en-male.mp4';
filmLink.dataset.i18n = 'watchVideo';
filmLink.textContent = t('watchVideo');
reportBanner.append(reportLabel, filmLink);
document.body.prepend(reportBanner);
