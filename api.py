from functions import *

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
	'title': LazyString(lambda: 'API Untuk Sentiment Analysis'),
	'version': LazyString(lambda: '0.0.1'),
	'description': LazyString(lambda: 'Dokumentasi API untuk Sentiment Analysis Menggunakan Neural Network dan LSTM Dengan Input Teks Manual dan Upload File CSV')
	},
	host = LazyString(lambda: request.host)
)
swagger_config = {
	"headers": [],
	"specs": [
		{
			"endpoint": 'docs',
			"route": '/docs.json'
		}
	],
	"static_url_path": '/flasgger_static',
	"swagger_ui": True,
	"specs_route": '/docs/',
	# "ui_params": {
	#	"operationsSorter": "method",
	#	"tagsSorter": "method"
	# }
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)


@swag_from("docs/nn_text_input.yml", methods=['POST'])
@app.route('/nn_text_input', methods=['POST'])
def nn_text():

	original_text = request.form.get('Text')

	return nn_text_process(original_text)

@swag_from("docs/nn_file_input.yml", methods=['POST'])
@app.route('/nn_file_input', methods=['POST'])
def nn_file():

	file = request.files['Upfile']

	return nn_file_process(file)

@swag_from("docs/lstm_text_input.yml", methods=['POST'])
@app.route('/lstm_text_input', methods=['POST'])
def lstm_text():

	original_text = request.form.get('Text')

	return lstm_text_process(original_text)

@swag_from("docs/lstm_file_input.yml", methods=['POST'])
@app.route('/lstm_file_input', methods=['POST'])
def lstm_file():

	file = request.files['Upfile']

	return lstm_file_process(file)

if __name__ == '__main__':
	app.run()
