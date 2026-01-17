from flask import Flask, request, jsonify
from backend import BackendClass

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
backend = BackendClass()

@app.route("/search")
def search():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    return jsonify(backend.search(query))

@app.route("/search_body")
def search_body():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    return jsonify(backend.search_body(query))

@app.route("/search_title")
def search_title():
    query = request.args.get('query', '')
    if not query: return jsonify([])
    return jsonify(backend.search_title(query))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)