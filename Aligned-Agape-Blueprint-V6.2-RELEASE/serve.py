
import http.server, socketserver
from pathlib import Path

ROOT = Path(__file__).parent
PORT = 8008

class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        import posixpath, urllib, os
        path = path.split('?',1)[0].split('#',1)[0]
        path = posixpath.normpath(urllib.parse.unquote(path))
        if path.startswith('/out'):
            return str(ROOT / 'out' / posixpath.normpath(path.replace('/out','').lstrip('/')))
        else:
            return str(ROOT / 'webui' / path.lstrip('/'))

if __name__ == "__main__":
    print(f"Serving Web-UI from {ROOT/'webui'} and OUT from {ROOT/'out'} on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
