# Third-party notices

The root MIT license applies to original project code. It does not replace the
licenses of datasets, models, libraries or material derived from them.

| Source | Upstream terms | Project use |
| --- | --- | --- |
| [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) — Qwen | Apache2.0, per model card | Frozen Base; weights not distributed |
| [Mobile Actions](https://huggingface.co/datasets/google/mobile-actions) — Google | [CC BY4.0](https://creativecommons.org/licenses/by/4.0/) | Normalized tool schemas, prompts and expected calls in the Tool review |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) — Google Research | [CC BY4.0](https://creativecommons.org/licenses/by/4.0/), per dataset card | Coding research source; raw/processed datasets omitted |
| [HumanEval](https://github.com/openai/human-eval) — OpenAI | [MIT notice](licenses/HumanEval-MIT.txt) | Function prompts and reference completions in the saved original Coding review |
| [Magicoder OSS Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) — Magicoder authors | Upstream card and record-level source licenses | Local expansion selected2,661 references associated with MIT sources; corpus not redistributed |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — hexgrad contributors | Apache2.0 | Local synthetic narration; weights not redistributed |
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) contributors | MIT | Local production tool; package not vendored |

Dataset-derived material was normalized, selected and combined with predictions
and project scoring. Changes include tool-schema type names and prompt/template
representations. These adaptations do not imply endorsement by upstream authors.
Mobile Actions review content is retained for error inspection, with attribution
to Google and the original dataset linked above.

Dependencies retain their own licenses. Research versions are listed in
`requirements.txt`; the public viewer uses the Python standard library. External
weights, original dataset downloads and bundled production tools are excluded.
