from flask import Flask, jsonify, render_template, request


# Build the application once; tests can inject a service without loading datasets.
def create_app(service=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024
    if service is None:
        from .rag_service import LunaRAGService
        service = LunaRAGService()
    app.extensions['rag_service'] = service

    @app.get('/')
    def index():
        return render_template('index.html', domains=service.domains,
            default_domain='techqa', metrics=service.metrics,
            research_findings=service.research_findings,
            strategy_options=service.strategy_options, examples=service.examples)

    @app.get('/health')
    def health():
        return jsonify(service.health())

    # Validate before retrieval or paid generation, including malformed JSON values.
    def arguments():
        payload = request.get_json(silent=True) if request.is_json else request.form
        if not payload or not hasattr(payload, 'get'):
            raise ValueError('Provide a question in a JSON object or form.')
        question = payload.get('question', '')
        if not isinstance(question, str) or not question.strip() or len(question) > 20000:
            raise ValueError('Question must contain 1–20,000 characters.')
        top_k = payload.get('top_k', 5)
        if isinstance(top_k, bool) or str(top_k) not in [str(i) for i in range(1, 11)]:
            raise ValueError('Top K must be an integer from 1 to 10.')
        generate = payload.get('generate', False)
        if isinstance(generate, str) and generate.lower() in ('true', 'false'):
            generate = generate.lower() == 'true'
        if not isinstance(generate, bool):
            raise ValueError('Generate must be true or false.')
        use_saved = payload.get('use_saved', False)
        if not isinstance(use_saved, bool):
            raise ValueError('Use saved must be true or false.')
        domain = payload.get('domain', 'techqa')
        mode = payload.get('mode', 'hybrid_rrf_static')
        if domain not in service.domains:
            raise ValueError('Unknown domain.')
        if mode not in [option['id'] for option in service.strategy_options]:
            raise ValueError('Unknown strategy.')
        return dict(question=question.strip(), domain=domain, mode=mode,
                    top_k=int(top_k), generate=generate, use_saved=use_saved)

    @app.post('/ask')
    def ask():
        try:
            return jsonify(service.answer(**arguments()))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception('Question request failed')
            return jsonify(error='Request failed. Inspect the server log.'), 500

    @app.post('/compare')
    def compare():
        try:
            values = arguments()
            values.pop('mode')
            return jsonify(service.compare(**values))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            app.logger.exception('Comparison failed')
            return jsonify(error='Comparison failed. Inspect the server log.'), 500

    return app
