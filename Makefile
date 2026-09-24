install:
	python -m pip install -r requirements.txt

train:
	python scripts/train_production_model.py

score:
	python scripts/score_customers.py

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py
