# Make all targets .PHONY
.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

SHELL = /usr/bin/env bash
USER_NAME = $(shell whoami)

SRC_FOLDER_TO_VALIDATE = src
LOG_FOLDER_TO_VALIDATE = logs
CONFIG_FOLDER_TO_VALIDATE = config
APP_FOLDER_TO_VALIDATE = app

MINIO_DOCKER_COMPOSE_YAML = docker-compose-minio.yaml
MLFLOW_MINIO_POSTGRES_DOCKER_COMPOSE_YAML = docker-compose.yaml
APP_DOCKER_COMPOSE_YAML = docker-compose-app-gpu.yaml
JENKINS_DOCKER_COMPOSE_YAML = docker-compose-jenkins.yaml

export

# Returns true if the stem is a non-empty environment variable, or else raises an error.
guard-%:
	@#$(or ${$*}, $(error $* is not set))

## Sort using isort
sort:
	poetry run isort --atomic $(SRC_FOLDER_TO_VALIDATE) $(LOG_FOLDER_TO_VALIDATE) $(CONFIG_FOLDER_TO_VALIDATE) $(APP_FOLDER_TO_VALIDATE) *.py

## Check sorting using isort
sort-check:
	poetry run isort --check-only --atomic $(SRC_FOLDER_TO_VALIDATE) $(LOG_FOLDER_TO_VALIDATE) $(CONFIG_FOLDER_TO_VALIDATE) $(APP_FOLDER_TO_VALIDATE) *.py

## Format using ruff
format:
	poetry run ruff format $(SRC_FOLDER_TO_VALIDATE) $(LOG_FOLDER_TO_VALIDATE) $(CONFIG_FOLDER_TO_VALIDATE) $(APP_FOLDER_TO_VALIDATE) *.py

## Check format using ruff
format-check:
	poetry run ruff format --check $(SRC_FOLDER_TO_VALIDATE) $(LOG_FOLDER_TO_VALIDATE) $(CONFIG_FOLDER_TO_VALIDATE) $(APP_FOLDER_TO_VALIDATE) *.py

## Lint code
lint: format sort
	poetry run ruff check --fix $(SRC_FOLDER_TO_VALIDATE) $(LOG_FOLDER_TO_VALIDATE) $(CONFIG_FOLDER_TO_VALIDATE) $(APP_FOLDER_TO_VALIDATE) *.py

## Check linting
lint-check: up format-check sort-check
	poetry run ruff check $(SRC_FOLDER_TO_VALIDATE) $(LOG_FOLDER_TO_VALIDATE) $(CONFIG_FOLDER_TO_VALIDATE) $(APP_FOLDER_TO_VALIDATE) *.py

## Clean ruff cache
clean-ruff-cache:
	poetry run ruff clean

## Clean logging file
clean-log:
	if [ -n "$$(ls logs/*.log 2>/dev/null)" ]; then \
		rm logs/*.log; \
	fi

## Down all docker container
down-all:
	docker compose -f ${MINIO_DOCKER_COMPOSE_YAML} -f ${MLFLOW_MINIO_POSTGRES_DOCKER_COMPOSE_YAML} -f ${JENKINS_DOCKER_COMPOSE_YAML} down

## Run Docker Compose to launch MinIO
minio: down-all
	docker compose -f ${MINIO_DOCKER_COMPOSE_YAML} up -d

## Run Docker Compose to set up MLflow for model tracking
mlflow: down-all
	docker compose -f ${MLFLOW_MINIO_POSTGRES_DOCKER_COMPOSE_YAML} up -d

## Version data using DVC for the first time
version-data-first:
	rm -rf .dvc/
	poetry run python task_version_data.py

## Version data using DVC
version-data:
	poetry run python task_version_data.py

## Run the training task locally
train: clean-log
	poetry run python task_train.py

## Download weight from minio to local folder:
download-weight:
	poetry run python task_download_weight_from_minio.py

## Open jupyter lab
jupyter:
	poetry run jupyter lab

## Deploy the FastAPI app locally
app-local:
	poetry run uvicorn app.application:app --host 0.0.0.0 --port 8000

## Deploy the FastAPI app to a Docker container
app-docker:
	docker compose -f ${APP_DOCKER_COMPOSE_YAML} up -d

## Run the whole pipeline
pipeline: mlflow clean-log
	poetry run python task_all.py

## Run Docker Compose to launch Jenkins
jenkins:
	docker compose -f ${JENKINS_DOCKER_COMPOSE_YAML} up -d

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=36 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
