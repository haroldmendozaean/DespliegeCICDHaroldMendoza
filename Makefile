install:
	pip install -r requirements.txt

train:
	python src/train.py

validate:
	python src/evaluate.py

test:
	pytest -q
