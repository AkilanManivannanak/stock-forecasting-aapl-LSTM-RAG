.PHONY: train eval serve test

train:
	python -m src.main_training

eval:
	python -m src.main_training

serve:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest


