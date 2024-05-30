.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = usl_for_embedding
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Make Dataset
data: test_environment
	$(PYTHON_INTERPRETER) src/data/make_dataset.py 

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif


## TDOO: Create the ray and mlflow server
## Set up python interpreter environment
create_environment:
	@echo "Creating or updating conda environment from environment.yml..."
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "Environment setup completed. Activate with:\nconda activate $(PROJECT_NAME)"




## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

METHOD ?= usl
MODE ?= train
DATA ?= zhaw



## Run main.py with specified method and mode (default: usl, train)
run:
	$(PYTHON_INTERPRETER) src/main.py --method $(METHOD) --mode $(MODE) --dataset $(DATA)

# ZHAW_Embedding_One
## Train models with usl method
train_usl_one:
	$(MAKE) run METHOD=usl MODE=train DATA=zhaw

## Evaluate models with usl method
eval_usl_one:
	$(MAKE) run METHOD=usl MODE=eval DATA=zhaw

## Test models with usl method
test_usl_one:
	$(MAKE) run METHOD=usl MODE=test DATA=zhaw

## Train models with usl-t method
train_usl_t_one:
	$(MAKE) run METHOD=usl-t MODE=train Data=zhaw

## Evaluate models with usl-t method
eval_usl_t_one:
	$(MAKE) run METHOD=usl-t MODE=eval DATA=zhaw

## Test models with usl-t method
test_usl_t_one:
	$(MAKE) run METHOD=usl-t MODE=test Data=zhaw


# ZHAW_Embedding_Two
## Train models with usl method
train_usl_two:
	$(MAKE) run METHOD=usl MODE=train DATA=zhaw_segments

## Evaluate models with usl method
eval_usl_two:
	$(MAKE) run METHOD=usl MODE=eval DATA=zhaw_segments

## Test models with usl method
test_usl_two:
	$(MAKE) run METHOD=usl MODE=test DATA=zhaw_segments

## Train models with usl-t method
train_usl_t_two:
	$(MAKE) run METHOD=usl-t MODE=train Data=zhaw_segments

## Evaluate models with usl-t method
eval_usl_t_two:
	$(MAKE) run METHOD=usl-t MODE=eval DATA=zhaw_segments

## Test models with usl-t method
test_usl_t_two:
	$(MAKE) run METHOD=usl-t MODE=test Data=zhaw_segments


# Curlie
## Train models with usl method
train_usl_curlie:
	$(MAKE) run METHOD=usl MODE=train DATA=curlie

## Evaluate models with usl method
eval_usl_curlie:
	$(MAKE) run METHOD=usl MODE=eval DATA=curlie

## Test models with usl method
test_usl_curlie:
	$(MAKE) run METHOD=usl MODE=test DATA=curlie

## Train models with usl-t method
train_usl_t_curlie:
	$(MAKE) run METHOD=usl-t MODE=train Data=curlie

## Evaluate models with usl-t method
eval_usl_t_curlie:
	$(MAKE) run METHOD=usl-t MODE=eval DATA=curlie

## Test models with usl-t method
test_usl_t_curlie:
	$(MAKE) run METHOD=usl-t MODE=test Data=curlie


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

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
		-v indent=19 \
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
