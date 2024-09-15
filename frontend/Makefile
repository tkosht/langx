default: 
	@echo "default target: do nothing."

# ==========
# frontend tasks
install:
	sh bin/install_vue.sh

init:
	sh bin/init_frontend.sh

restore: ci

abc:
	$(eval task=$(shell echo "$@" | perl -pe 's/frontend-//'))
	@echo $(task)

ci:
	npm ci

prod:
	npm run build

dev:
	HOST=0.0.0.0 npm run dev

unit:
	npm run unit

e2e:
	rm -rf user_data
	npm run e2e

test-unit: ci unit

test-e2e: ci prod e2e

debug-e2e:
	npm run debug-e2e
