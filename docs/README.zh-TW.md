# Adaptive LoRA Lab

**小型專用 adapter、共用 Base、可檢查的實驗證據。**

[English README](../README.md) · [男聲介紹影片](../media/adaptive-lora-lab-star-en-male.mp4)

這是已完成的本機研究與作品集專案，探索如何在凍結的 Qwen3-8B-Base 上，
透過 Tool-use 和 Coding LoRA 改善特定任務，再以透明的規則路由整合。

## STAR 流程與成果

- **Situation：** 流暢文字不代表工具呼叫或程式碼正確，本機 GPU 資源也有限。
- **Task：** 建立可切換的專用 adapter，保留測試資料，做可重現的比較與 demo。
- **Action：** 資料整理、QLoRA 訓練、格式失敗診斷與修正、rank／資料量／掛載位置
  比較、重複 seeds、共用 Base 切換、規則路由及獨立 CPU 重播／重新評分。
- **Result：** Tool 在961題中893題完整呼叫正確（92.92%）；擴充 Coding 在已知
  HumanEval164 的通過數由 Base 的98增至111；三個 Coding seeds 在同一465題
  MBPP 子集平均為53.62%；Router 通過27/27路由契約。

Coding 初版的97/164未超越 Base，這個負面結果有保留。Router 的22/24對14/24
是小規模整合測試，兩者各含5個釐清案例。這些數字不代表普遍能力或正式產品認證。

## 無需 GPU 即可瀏覽

Python3.11以上，在根目錄執行：

```bash
python tools/serve_reports.py
```

開啟 [本機成果頁](http://127.0.0.1:18765/#pipeline)，可看14階段流程、成果表格、
68個 Tool 差異案例、原始 Coding 的164題 review，以及 CPU 路由預覽。
介面支援英文和繁體中文。此入口不下載模型，也不產生新模型回覆。

## 導覽

- [`docs/`](.)：架構、重現指南與歷史研究紀錄。
- [`reports/`](../reports/README.md)：精簡報告、來源雜湊與 UI 快照。
- [`media/`](../media/README.md)：3分28秒英文男聲影片、字幕、旁白稿與封面。
- `training/`、`experiments/`、`evaluation/`：保留的訓練、實驗和評估程式。
- `runtime/`、`demo/`：共用模型、mock tools、路由及本機介面。

大型模型、原始／處理後資料、checkpoint、虛擬環境和完整輸出保留在本機，不加入 Git。
公開報告可供查看，但不能取代完整原始檔的獨立重播。

原創程式碼採 [MIT](../LICENSE)；第三方授權見 [來源說明](../THIRD_PARTY_NOTICES.md)。
