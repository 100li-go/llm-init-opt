.PHONY: smoke select strategies run analyze test all

smoke:
	python scripts/00_smoke_test.py

select:
	python scripts/o1_select_problems.py

strategies:
	python scripts/o2_generate_strategies.py

run:
	mkdir -p logs
	nohup python scripts/o3_run_experiments.py > logs/run.log 2>&1 &
	@echo "Running in background, check logs/run.log"

analyze:
	python scripts/o4_analyze.py

test:
	pytest tests/ -v

all: select strategies run analyze
