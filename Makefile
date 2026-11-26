PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: all venv data part1 part2 part3 test clean-results

all: part1 part2 part3 test

venv:
	uv venv -p 3.11 .venv && uv pip install -p .venv -r requirements.txt

data/raw/invocations_per_function_md.anon.d01.csv:
	mkdir -p data/raw
	curl -L -o data/raw/trace.tar.xz https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz
	tar -xJf data/raw/trace.tar.xz -C data/raw --wildcards 'invocations_per_function_md*'
	rm data/raw/trace.tar.xz

data: data/raw/invocations_per_function_md.anon.d01.csv
	$(PY) src/prepare_data.py

part1:
	$(PY) src/part1_optimizers.py

part2: part1
	$(PY) src/part2_capacity.py

part3: part1
	$(PY) src/part3_simulation.py

test:
	$(PY) -m pytest -q tests
