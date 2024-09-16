default: all

all: up


# ==========
# interaction tasks
bash:
	docker compose exec app bash

poetry demo demo-stream demo-code-interpreter campfire-data:
	@make backend-$@

python: up
	docker compose exec app python

# lowcode:
# 	docker compose up lowcode_llm

# switch mode
cpu gpu:
	@rm -f compose.yml
	@ln -s docker/compose.$@.yml compose.yml

db:
	@rm -f compose.override.yml
	@ln -s docker/compose.override.$@.yml compose.override.yml

mode:
	@echo $$(ls -l compose.yml | awk -F. '{print $$(NF-1)}')

mode-nooverride:
	@rm -f compose.override.yml


# ==========
# general tasks
pip: _pip commit

_pip:
	docker compose exec app python -m pip install --user -U -r requirements.txt         # too slow

commit:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	docker commit experiment.app experiment.app:latest
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

save: commit
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	docker save experiment.app:latest | gzip > data/experiment.app.tar.gz
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

load:
	@echo "$$(date +'%Y/%m/%d %T') - Start $@"
	docker load < data/experiment.app.tar.gz
	@echo "$$(date +'%Y/%m/%d %T') - End $@"

# ==========
# docker compose aliases
up:
	docker compose up -d

active:
	docker compose up

ps images down:
	docker compose $@

im: images

build:
	docker compose build

build-no-cache:
	docker compose build --no-cache

reup: down up

clean: clean-container clean-cache

clean-container:
	docker compose down --rmi all

clean-cache:
	rm -rf local_cache/ __pycache__

