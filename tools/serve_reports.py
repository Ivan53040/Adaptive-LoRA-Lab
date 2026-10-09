"""Serve the published report snapshot and real CPU route previews. Standard library only."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.router import route_request

ASSETS = {f'/{name}': ROOT / 'demo' / name for name in
          ('index.html', 'app.js', 'i18n.js', 'pipeline.js', 'coding.js', 'style.css')}
ASSETS['/report-mode.js'] = ROOT / 'tools/report-mode.js'
for name in ('adaptive-lora-lab-star-en-male.mp4', 'cover.jpg', 'adaptive-lora-lab-en.srt'):
    ASSETS[f'/media/{name}'] = ROOT / 'media' / name
API_FILES = {'/api/status': 'status.json', '/api/review': 'tool-review.json',
             '/api/coding-review': 'coding-review.json'}
MIMES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
         '.css': 'text/css; charset=utf-8', '.mp4': 'video/mp4', '.jpg': 'image/jpeg',
         '.srt': 'text/plain; charset=utf-8'}

def snapshot():
    value = deepcopy(REPORTS['/api/status'])
    value['now'] = datetime.now(timezone(timedelta(hours=10))).strftime('%Y-%m-%dT%H:%M:%S')
    return value

REPORTS = {url: json.loads((ROOT / 'reports/demo' / name).read_text(encoding='utf-8'))
           for url, name in API_FILES.items()}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, value, mime='application/json; charset=utf-8', head=False):
        payload = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        path = urlsplit(self.path).path
        if path == '/api/status':
            return self.reply(200, snapshot(), head=head)
        if path in REPORTS:
            return self.reply(200, REPORTS[path], head=head)
        if path == '/api/tools':
            return self.reply(200, (ROOT / 'configs/mobile_action_tools.json').read_bytes(), head=head)
        if path == '/':
            path = '/index.html'
        if path not in ASSETS:
            return self.reply(404, {'error': 'Not found.'}, head=head)
        file = ASSETS[path]
        payload = file.read_bytes()
        if path == '/index.html':
            payload = payload.replace(b'<script src="/i18n.js"></script>',
                b'<script src="/i18n.js"></script><script src="/report-mode.js"></script>')
        self.reply(200, payload, MIMES[file.suffix], head=head)

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ('/api/route', '/api/reset'):
            return self.reply(403, {'error': 'This report viewer does not load models or execute generated programs.'})
        try:
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Use application/json.')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                raise ValueError('Invalid request size.')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError('The request must be a JSON object.')
            if path == '/api/reset':
                return self.reply(200, snapshot())
            if request.get('generate', False) is not False:
                raise ValueError('Only route previews are available in this report viewer.')
            choice = request.get('coding_variant', 'pilot')
            if choice not in ('pilot', 'expanded'):
                raise ValueError('Coding variant must be pilot or expanded.')
            route = route_request(request.get('prompt'), request.get('allow_experimental_coding', False))
            route['coding_variant'] = choice if route['adapter'] == 'coding' else None
            self.reply(200, {'route': route, 'kind': 'clarify' if route['domain'] == 'clarify' else 'preview',
                             'state': snapshot()})
        except (ValueError, TypeError) as error:
            self.reply(400, {'error': str(error)})

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--port', type=int, default=18765)
    args = cli.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(f'Reports: http://127.0.0.1:{args.port}/#pipeline', flush=True)
        print('Saved results and CPU route previews. No model, GPU, Docker or downloads required.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
