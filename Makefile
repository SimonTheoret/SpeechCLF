cleandf: preprocessing/data_processing.py
	python3 -m spacy download en_core_web_sm
	python3 -m preprocessing.data_processing

cleandfnocorrect: preprocessing/data_processing.py
	python3 -m preprocessing.data_processing --dest=data/cleanedNoCorrect.csv --corrector=None

profile: eda/profiling.py
	python3 -m eda.profiling

test: tests
	python3 -m pytest -rP
