default: all

all: up install

install: backend-poetry-install

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
	docker compose exec app python -m pip install -r requirements.txt         # too slow

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

ssh:
	docker compose exec app sudo service ssh start

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

clean: clean-container backend-clean clean-logs

clean-all: clean clean-database

clean-logs:
	rm -rf log/*.log

clean-database:
	rm -rf data/postgres

clean-graphdb:
	sudo rm -rf data/neo4j

clean-container:
	docker compose down --rmi all
	rm -rf app/__pycache__

# ==========
# frontend tasks
frontend-install frontend-init frontend-ci frontend-prod frontend-dev frontend-unit frontend-e2e : up
	$(eval task_name=$(shell echo "$@" | perl -pe 's/frontend-//'))
	@echo "runnning task @ frontend: $(task_name)"
	docker compose exec app sudo service dbus start
	docker compose exec app bash -c "cd frontend && make $(task_name)"

frontend-restore: frontend-ci

# ==========
# backend tasks
backend-demo backend-poetry-install backend-poetry backend-demo-stream backend-demo-code-interpreter: up
	$(eval task_name=$(shell echo "$@" | perl -pe 's/backend-//'))
	@echo "runnning task @ backend: $(task_name)"
	docker compose exec app bash -c "cd backend && make $(task_name)"

backend-campfire-data backend-ls: up
	$(eval task_name=$(shell echo "$@" | perl -pe 's/backend-//'))
	@echo "runnning task @ backend: $(task_name) with poetry"
	docker compose exec app bash -c 'cd backend && $$HOME/.local/bin/poetry run make $(task_name)'


backend-clean:
	$(eval task_name=$(shell echo "$@" | perl -pe 's/backend-//'))
	@echo "runnning task @ backend: $(task_name)"
	cd backend && make $(task_name)
