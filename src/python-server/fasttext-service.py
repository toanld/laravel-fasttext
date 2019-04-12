# FastText Flask Web Service

import fastText
import json
from flask import Flask, jsonify, abort, request, send_from_directory, render_template, g
import os
import re

app = Flask(__name__)

default_data = {}
default_data['web64'] = {
	'app': 'laravel-fasttext',
	'version':	'0.0.2',
	'last_modified': '2019-04-12',
	'documentation': 'http://nlpserver.web64.com/',
	'github': 'https://github.com/web64/nlp-server',
	'endpoints': ['/predict', '/models'],
}

@app.route("/")
def main():
	default_data['web64']['available_models'] = list(g.fasttext_models.keys())
	# default_data['web64']['available_models'] = {} #list(g.fasttext_models.keys())

	# with open('fasttext-models.json') as json_file:
	# 	data = json.load(json_file)
	# 	#default_data['web64']['json'] = data
	# 	for model, path in data['models'].items():
	# 		default_data['web64']['available_models'][model] = path

	return jsonify(default_data)

@app.route("/models", methods=["GET", "POST"], endpoint="models")
def models():
	response_data = {}
	response_data['models'] = {}
	#response_data['model_settings'] = {}

	with open('fasttext-models.json') as json_file:
		data = json.load(json_file)
		for model, path in data['models'].items():
			#response_data['models'][model] = path
			model_args = g.fasttext_models[model].f.getArgs()
			response_data['models'][model] = {
				'path': path,
				"lr": model_args.lr,
				"lrUpdateRate": model_args.lrUpdateRate,
				"epoch": model_args.epoch,
				"dim": model_args.dim,
				"ws": model_args.ws,
				"model": str(model_args.model)[len("model_name."):],
				"loss": str(model_args.loss)[len("loss_name."):],
				"wordNgrams": model_args.wordNgrams,
				"minCountLabel": model_args.minCountLabel,
				"label": model_args.label,
				"thread": model_args.thread,
				"bucket": model_args.bucket,
				"cutoff": model_args.cutoff,
				"t": model_args.t,
				"minn": model_args.minn,
				"maxn": model_args.maxn,
				"isQuant": g.fasttext_models[model].f.isQuant()
			}

	return jsonify(response_data)

@app.route("/predict", methods=["GET", "POST"], endpoint="predict")
def predict():
    """
       Retrieve predictions for a single word or sentence from the deployed model.
       Query String:
          * q (str):  word or sentence to get a vector representation for.
          * limit (int):  Number of most likely classes returned (default: 1)
          * threshold (float): Filter classes with a probability below threshold (default: 0.0)
       Returns:
           A json containing the vector representations.
    """

    limit = int(request.args.get('limit')) if request.args.get('limit') else 2
    threshold = float(request.args.get('threshold')) if request.args.get('threshold') else 0.0
    model = str(request.args.get('model')) if request.args.get('model') else 'default'
    query = request.args.get('q')

    if model not in g.fasttext_models:
        return jsonify({
    	    'model': model,
    	    'status': 'ERROR',
    	    'message': 'Model not found',
        	'data': {}
        	})

    res = make_prediction(query, model, limit, threshold)
    print('results:')
    print(res)
    return jsonify({
    	'model': model,
    	'status': g.fasttext_status,
    	'message': g.fasttext_message,
    	'data': res
    	})

@app.before_request
def before_request():

	# g.fasttext_models = {
	# 	'default' : fastText.load_model('/home/forge/web64/nlp/reactions/model_storyboard.bin'),
	# 	'no-reactions' : fastText.load_model('/home/forge/web64/nlp/reactions/model_storyboard.bin'),
	# 	'no-shares' : fastText.load_model('/home/forge/web64/nlp/shares/model_storyboard.bin'),
	# 	'no-domains' : fastText.load_model('/home/forge/web64/nlp/domain/model_storyboard.bin'),
	# 	'no-vip': fastText.load_model('/home/forge/web64/nlp/vip_300docs/model_storyboard.bin')
	# }

	g.fasttext_status = 'OK'
	g.fasttext_message = ''

	g.fasttext_models = {}
	with open('fasttext-models.json') as json_file:
		data = json.load(json_file)
		#default_data['web64']['json'] = data
		for model, path in data['models'].items():
			g.fasttext_models[model] = fastText.load_model(path)

	print('loaded models..')


def make_prediction(q, model, limit, threshold):
	try:
		labels, probabilities = g.fasttext_models[model].predict(q, limit, threshold)
	except:
		g.fasttext_status = 'ERROR'
		g.fasttext_message = 'No prediction possible'
		return {}

	# if not probabilities:
	# 	return jsonify({
	# 		'model': model,
	# 		'status': 'ERROR',
	# 		'message': 'No prediction possible',
	# 		'data': ''
	# 	})

	return [{"label": l.replace('__label__', ''), "probability": p, "percent": round((p*100), 2)} for l, p in zip(labels, probabilities)]



app.run(host='0.0.0.0', port=6410, debug=False)