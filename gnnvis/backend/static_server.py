from http.server import HTTPServer, SimpleHTTPRequestHandler, test as test_orig
import sys
from functools import partial


def test(*args):
    test_orig(*args, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)


if __name__ == '__main__':
    Handler = partial(CORSRequestHandler, directory='backend')
    test(Handler, HTTPServer)
