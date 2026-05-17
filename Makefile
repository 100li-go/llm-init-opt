.PHONY: smoke select candidates run analyze test all

smoke:
	PYTHONPATH=/mnt/d/project2 python3 scripts/00_smoke_test.py

select:
	PYTHONPATH=/mnt/d/project2 python3 scripts/o1_select_problems.py

candidates:
	PYTHONPATH=/mnt/d/project2 python3 scripts/o2_generate_candidates.py


run:
	mkdir -p logs
	cd /mnt/d/project2 && PYTHONPATH=/mnt/d/project2 nohup python3 scripts/o3_run_experiments.py > logs/run.log 2>&1 &
	@echo "Running in background, check logs/run.log"

analyze:
	PYTHONPATH=/mnt/d/project2 python3 scripts/o4_analyze.py

test:
	python3 -m pytest tests/ -v

all: select candidates run analyze
