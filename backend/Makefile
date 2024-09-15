# backend tasks
exec_date := $(shell date '+%Y%m%d%H%M%S')


# ==========
# environments
poetry-install:
	@sh bin/poetry_install.sh

poetry:
	@bash -i -c 'SHELL=/usr/bin/bash poetry shell'

ls:
	ls

# ==========
# # for app
demo:
	@bash -i -c 'SHELL=/usr/bin/bash poetry run bash bin/run_demo.sh'

demo-stream:
	@bash -i -c 'SHELL=/usr/bin/bash poetry run bash bin/run_demo_stream.sh'

demo-code-interpreter:
	@bash -i -c 'SHELL=/usr/bin/bash poetry run bash bin/run_demo_code_interpreter.sh'

tool-maker:
	python -m app.llm_toolmaker.tool_maker --task-name=word_sorting

eval:
	python -m app.llm_toolmaker.eval --task-name=word_sorting

summary-semantic-memory:
	python -m app.semantic_kernel.component.semantic_memory

demo-semantic-memory:
	python -m app.semantic_kernel.executable.demo

campfire-data: campfire-data-popular campfire-data-fresh campfire-data-last_spurt campfire-data-most_funded campfire-data-density campfire-data-to-rdb

campfire-data-popular:
	python -m app.business.campfire.downloader --execution-id=$(exec_date) --max-pages=5 --sortby=popular

campfire-data-fresh:
	python -m app.business.campfire.downloader --execution-id=$(exec_date) --max-pages=5 --sortby=fresh

campfire-data-last_spurt:
	python -m app.business.campfire.downloader --execution-id=$(exec_date) --max-pages=5 --sortby=last_spurt

campfire-data-most_funded:
	python -m app.business.campfire.downloader --execution-id=$(exec_date) --max-pages=5 --sortby=most_funded

campfire-data-density:
	python -m app.business.campfire.downloader --execution-id=$(exec_date) --max-pages=5 --sortby=density

campfire-data-to-rdb:
	python -m app.business.campfire.to_rdb



# ==========
# # webapi tasks
webapi:
	uvicorn \
        --host=0.0.0.0 \
        --log-config=conf/logging.ini \
        --app-dir=. \
        webapi:app

test-unit ut:
	python -m pytest test

tail-log:
	@tail -0f log/app.log

hello-request:
	@sh bin/request_hello.sh

post-request:
	@sh bin/request_post.sh

test-request:
	@sh bin/test_request.sh
# ==========
# # docker task
bash:
	docker exec -it experiment.app bash

# ==========
# # mlflow tasks
mlflow mlflow-server:
	mlflow server --host=0.0.0.0 \
		--backend-store-uri sqlite:///data/mlflow.db \
		--default-artifact-root=mlruns

mlflow-notebook:
	mlflow server --host=0.0.0.0 \
		--backend-store-uri sqlite:///./data/experiment.db \
		--default-artifact-root=mlruns

mlflow-ui:
	mlflow ui --host=0.0.0.0

sqlite-notebook:
	sqlite_web --port=5656 --host=0.0.0.0 ./data/experiment.db

sqlite-newsdata:
	sqlite_web --port=5656 --host=0.0.0.0 ./data/scrapy_data/news.db


# ==========
# # tensorboard tasks
tensorboard:
	$(eval logdir:=$(shell ls -trd result/* | egrep -v db | tail -n 1))
	echo $(logdir)
	tensorboard --host=0.0.0.0 --logdir=$(logdir)

# # GPU tasks
nvidia nvidia-smi:
	nvidia-smi \
	  --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used \
	  --format=csv \
	  -l 1

# ==========
# clean tasks
clean-pycache:
	find . -name '__pycache__' | xargs rm -rf

clean-result:
	rm -rf result/*

clean-mlruns:
	sh bin/clean_mlruns.sh

clean-venv:
	rm -rf .venv/
	rm -f poetry.lock

_clean-experiment:	# if use, be carefull !!
	rm -rf mlruns data/mlflow.db

# clean: clean-result clean-mlruns clean-venv
clean: clean-result clean-venv

