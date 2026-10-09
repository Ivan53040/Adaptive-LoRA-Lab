"""Check a clean public checkout without loading models or requiring research artifacts."""
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = read('reports/manifest.json')
    for artifact in manifest['artifacts']:
        path = ROOT / artifact['path']
        require(path.is_file(), f'Missing published artifact: {path.name}')
        require(digest(path) == artifact['published_sha256'], f'Published artifact changed: {path.name}')
        require(path.stat().st_size < 100 * 1024**2, f'File exceeds GitHub regular-file limit: {path.name}')
    for name, expected in manifest['frozen_sources_and_configs'].items():
        require('\\' not in name, f'Non-portable path in public manifest: {name}')
        require(digest(ROOT / name) == expected, f'Frozen research file changed: {name}')
    print(f'PASS: {len(manifest["artifacts"])} published artifacts and {len(manifest["frozen_sources_and_configs"])} frozen fingerprints.')

    final = read('reports/final-summary.json')
    tool = read('reports/tool-test-summary.json')
    status = read('reports/demo/status.json')
    require(final['completed'] is True, 'Final report is incomplete.')
    require([x['passed'] for x in final['known_humaneval']] == [98, 97, 111], 'HumanEval counts differ.')
    require([x['test_passed'] for x in final['coding_seeds']] == [249, 252, 247], 'Coding seed counts differ.')
    require(tool['examples'] == 961 and round(tool['metrics']['full_call_exact_match_rate'] * 961) == 893, 'Tool counts differ.')
    require(status['benchmark']['lora'] == tool['metrics'], 'Viewer and report Tool metrics differ.')
    require(len(read('reports/demo/tool-review.json')) == 68, 'Tool review coverage differs.')
    require(len(read('reports/demo/coding-review.json')) == 164, 'Coding review coverage differs.')
    require(status['status'] == 'paused' and status['shared_base_load_count'] == 0, 'Snapshot must be report-only.')

    from runtime.router import route_request
    cases = read('data/evaluation/router-expanded-v1.json')['cases']
    for case in cases:
        route = route_request(case['prompt'], case.get('allow_experimental_coding', False))
        require((route['domain'], route['adapter']) == (case['domain'], case['adapter']),
                f'Route contract failed: {case["id"]}')
    require(len(cases) == 27, 'Expected the authored27-case routing contract set.')
    print('PASS: reported counts, saved review coverage and27 real CPU route contracts.')

    markdown = [ROOT/'README.md', ROOT/'THIRD_PARTY_NOTICES.md']
    for directory in ('docs', 'reports', 'media'):
        markdown.extend((ROOT/directory).rglob('*.md'))
    for file in markdown:
        text = re.sub(r'```.*?```', '', file.read_text(encoding='utf-8'), flags=re.S)
        for target in re.findall(r'\]\(([^)]+)\)', text):
            target = target.strip().strip('<>')
            if urlsplit(target).scheme or target.startswith('#'):
                continue
            relative = unquote(target.split('#', 1)[0].split('?', 1)[0])
            require((file.parent/relative).exists(), f'Broken local link in {file.relative_to(ROOT)}: {target}')
    print('PASS: local documentation and media links.')

    from tools.serve_reports import Handler
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(path, body=None, expected=200):
        data = None if body is None else json.dumps(body).encode()
        req = Request(base+path, data=data, headers={'Content-Type':'application/json'})
        try:
            with urlopen(req, timeout=10) as response:
                code, payload = response.status, response.read()
        except HTTPError as error:
            code, payload = error.code, error.read()
        require(code == expected, f'{path}: expected HTTP{expected}, got{code}')
        return payload
    try:
        require(b'report-mode.js' in request('/'), 'Portable labels missing from HTML.')
        require(json.loads(request('/api/status'))['status'] == 'paused', 'Viewer must not load a model.')
        require(len(json.loads(request('/api/review'))) == 68, 'Tool HTTP review coverage differs.')
        require(len(json.loads(request('/api/coding-review'))) == 164, 'Coding HTTP review coverage differs.')
        result = json.loads(request('/api/route', {'prompt':'Please turn on the flashlight.', 'generate':False}))
        require((result['kind'], result['route']['adapter']) == ('preview', 'tool'), 'HTTP route preview differs.')
        result = json.loads(request('/api/route', {'prompt':'def square(x):\n', 'generate':False,
                      'allow_experimental_coding':True, 'coding_variant':'expanded'}))
        require(result['route']['coding_variant'] == 'expanded', 'Expanded Coding opt-in differs.')
        request('/api/run', {'prompt':'hello'}, 403)
        request('/api/code', {'prefix':'def f():\n'}, 403)
        request('/api/route', {'prompt':'hello', 'generate':True}, 400)
        request('/outputs/private-file.json', expected=404)
        request('/../README.md', expected=404)
        for asset in ('app.js', 'i18n.js', 'pipeline.js', 'coding.js', 'style.css', 'report-mode.js'):
            require(len(request('/'+asset)) > 0, 'Empty viewer asset: '+asset)
        clip = request('/media/adaptive-lora-lab-star-en-male.mp4')
        require(clip[4:8] == b'ftyp', 'Media endpoint did not return an MP4.')
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    print('PASS: portable HTTP viewer, media, previews and generation/file-access boundaries.')

if __name__ == '__main__':
    main()
