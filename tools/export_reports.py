"""Export selected completed local artifacts; never train or load model weights."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = {
    'outputs/gpu-remaining-20261007/final-summary.md': 'reports/final-summary.md',
    'outputs/gpu-remaining-20261007/final-summary.json': 'reports/final-summary.json',
    'outputs/data-size-r8/comparison.md': 'reports/tool-data-size.md',
    'outputs/data-size-r8/comparison.json': 'reports/tool-data-size.json',
    'outputs/data-size-r8/3k/evaluation/test-summary.json': 'reports/tool-test-summary.json',
    'outputs/coding-expansion-3k/comparison.md': 'reports/coding-expansion.md',
    'outputs/coding-followup/comparison.md': 'reports/coding-followup.md',
    'outputs/router-expanded-v1/comparison.md': 'reports/router-integration.md',
    'outputs/router-expanded-v1/comparison.json': 'reports/router-integration.json',
    'outputs/portfolio-video/Adaptive-LoRA-Lab-STAR-English-Male.mp4': 'media/adaptive-lora-lab-star-en-male.mp4',
    'outputs/portfolio-video/Adaptive-LoRA-Lab-English-Male.srt': 'media/adaptive-lora-lab-en.srt',
    'outputs/portfolio-video/Narration-English-Male.txt': 'media/narration-en.txt',
    'outputs/portfolio-video/Adaptive-LoRA-Lab-Poster.jpg': 'media/cover.jpg',
    'outputs/expanded-demo-preview.png': 'media/shared-base-demo.png',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def public_value(value):
    if isinstance(value, dict):
        return {public_value(k): public_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str):
        # Keep relative evidence identities while removing workstation locations.
        for prefix in (str(ROOT) + '\\', ROOT.as_posix() + '/'):
            value = value.replace(prefix, '')
        for prefix in (str(Path.home()), Path.home().as_posix()):
            value = value.replace(prefix, '[LOCAL_HOME]')
    return value

def main():
    missing = [p for p in SOURCES if not (ROOT / p).is_file()]
    if missing:
        raise SystemExit('Export needs the completed local archive. Missing: ' + ', '.join(missing))
    from demo.expanded_server import ExpandedDemo
    from demo.server import DEFAULT_DEMO_ADAPTER
    app = ExpandedDemo(DEFAULT_DEMO_ADAPTER)
    app.status = 'paused'
    records = []
    for source, destination in SOURCES.items():
        src, dst = ROOT / source, ROOT / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == '.json':
            value = public_value(json.loads(src.read_text(encoding='utf-8')))
            dst.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        else:
            shutil.copyfile(src, dst)
        records.append({'path': destination, 'source': source,
                        'source_sha256': digest(src), 'published_sha256': digest(dst)})
    snapshot = app.snapshot()
    snapshot['now'] = '2026-10-09T12:00:00'
    exports = {'status.json': snapshot, 'tool-review.json': app.review,
               'coding-review.json': app.coding_review['cases']}
    for name, value in exports.items():
        dst = ROOT / 'reports/demo' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(public_value(value), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        records.append({'path': dst.relative_to(ROOT).as_posix(),
                        'source': 'Derived through the preserved ExpandedDemo report reader',
                        'published_sha256': digest(dst)})
    protocol = json.loads((ROOT / 'outputs/gpu-remaining-20261007/study_protocol.json').read_text())
    frozen = {**protocol['sources'], **protocol['configs']}
    for name, expected in frozen.items():
        if digest(ROOT / name) != expected:
            raise SystemExit('Frozen source/config changed: ' + name)
    manifest = {'exported_on': '2026-10-09', 'scope': 'Curated report snapshot, not the complete raw research archive',
                'transforms': ['Absolute workstation paths in JSON made relative or redacted',
                               'Saved UI state exported in report-only mode; no generation'],
                'artifacts': records, 'frozen_sources_and_configs': frozen}
    (ROOT / 'reports/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(records)} artifacts; {len(frozen)} frozen source/config fingerprints unchanged.')

if __name__ == '__main__':
    main()
