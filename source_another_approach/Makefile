
.PHONY: install train eval serve test format

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	PYTHONPATH=src python -m shop_ml.train --data_dir ./dataset --out_dir ./artifacts

eval:
	PYTHONPATH=src python -m shop_ml.eval --data_dir ./dataset --ckpt ./artifacts/best_model.pt --out_dir ./artifacts

serve:
	PYTHONPATH=src uvicorn shop_ml.serve:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=src pytest -q

format:
	python -m pip install ruff==0.5.5 && ruff check src --fix
