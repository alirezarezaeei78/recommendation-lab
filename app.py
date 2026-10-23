"""Loopback-only, read-only local recommendation explorer."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from recommender import Recommender, synthetic_rows, read_rows, csv_export
from bpr import BPRRecommender

ROOT = Path(__file__).resolve().parent

def handler_for(model, synthetic):
    class Handler(BaseHTTPRequestHandler):
        def local_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')
        def reply(self, data, kind='application/json', status=200):
            raw = data.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind+'; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            if not self.local_host():
                return self.reply(json.dumps({'error':'Loopback host required'}), status=403)
            path = urlparse(self.path).path
            assets = {'/': ('web/index.html', 'text/html'), '/app.js': ('web/app.js', 'application/javascript'), '/styles.css': ('web/styles.css', 'text/css')}
            if path in assets:
                f, mime = assets[path]; return self.reply((ROOT/f).read_text(encoding='utf-8'), mime)
            if path == '/api/catalog':
                return self.reply(json.dumps({'items': model.items, 'synthetic': synthetic,
                    'neighbors': model.neighbors, 'shrinkage': model.shrinkage,
                    'users': model.users if hasattr(model, 'user_factors') else [],
                    'bpr_training': getattr(model, 'training', None)}))
            self.reply(json.dumps({'error': 'Not found'}), status=404)

        def do_POST(self):
            try:
                if not self.local_host():
                    return self.reply(json.dumps({'error':'Loopback host required'}), status=403)
                if self.path != '/api/recommend':
                    return self.reply(json.dumps({'error':'Not found'}), status=404)
                host = self.headers.get('Host', '')
                if self.headers.get('Origin') not in (None, 'http://'+host):
                    return self.reply(json.dumps({'error':'Cross-origin request rejected'}), status=403)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    return self.reply(json.dumps({'error':'Request must be1..65536 bytes'}), status=413)
                args = json.loads(self.rfile.read(size))
                if not isinstance(args, dict) or set(args)-{'liked','seen','method','n','user_id'}:
                    raise ValueError('Unknown request fields')
                if args.get('user_id') is not None and not isinstance(args['user_id'], str):
                    raise ValueError('Training-user ID must be a string')
                for key in ('liked','seen'):
                    value = args.get(key, [])
                    if not isinstance(value, list) or not all(isinstance(x,str) for x in value):
                        raise ValueError('Item lists must contain strings')
                result = model.recommend(**args)
                result['csv'] = csv_export(result)
                self.reply(json.dumps(result, allow_nan=False))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.reply(json.dumps({'error':str(exc)}), status=400)
    return Handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Your appropriately licensed CSV: user,item,rating,timestamp')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--bpr-seed', type=int, default=13, help='Fixed default seed; not chosen from test results')
    args = parser.parse_args()
    model = BPRRecommender(read_rows(args.data) if args.data else synthetic_rows(), seed=args.bpr_seed)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(model, not args.data))
    print(f'Open http://127.0.0.1:{server.server_port}; local processing; synthetic={not bool(args.data)}', flush=True)
    server.serve_forever()
