# laravel-fasttext
Laravel Package for accessing FastText classification models


# Getting Started
The PHP and Laravel package requires a running fastText Web Service.

First install fastText for Python if not already installed:
```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```

Rename `src/python-server/fasttext-models.sample.json` to `fasttext-models.json` and list your trained FastText models there.
```json
{
"models": {
    "default": "/home/test/models/sentiment-analysis.bin",
    "sentiment-analysis": "/home/test/models/sentiment-analysis.bin",
    "sentiment-analysis-fr": "/home/test/models/sentiment-analysis-french.bin",
	}
}
```

To run the fastText web service run:
```bash
python3 fasttext-service.py
```
By default this will start the web service running on port 6410.

Check http://localhost:6410/ to see if the server is running and your models are lited.

Detailed info about the models can be seen here: http://localhost:6410/models

## Predictions
You can now get predictions from a specified model with any given text:

http://localhost:6410/predict?q=this+is+great+and+fantastic+news&model=sentiment-analysis

If no model is specifiec it will look for a model named `default`

## PHP/Laravel Package coming soon!

# Credits
A big thanks to dfederschmidt as we've borrowed a lot from fasttext-server:
https://github.com/dfederschmidt/fasttext-server

