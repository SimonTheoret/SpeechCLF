cleandf: preprocessing/data_processing.py
	python3 -m spacy download en_core_web_sm
	python3 -m preprocessing.data_processing

test: tests
	python3 -m pytest
