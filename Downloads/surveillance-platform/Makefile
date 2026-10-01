COMPOSE = docker compose --env-file .env -f infra/docker-compose.yml

.PHONY: help env up up-cams down reset ps logs check data data-verify test test-infra mot-video

help:
	@echo "make env          create .env from .env.example"
	@echo "make up           start postgres, redis, opensearch, mediamtx"
	@echo "make up-cams      same, plus fake camera cam1"
	@echo "make check        verify every service is healthy"
	@echo "make data         download / extract starter datasets"
	@echo "make data-verify  report which datasets are in place"
	@echo "make test         offline tests"
	@echo "make test-infra   tests that need the services running"
	@echo "make mot-video    turn one MOT17 sequence into data/sample_videos/sample.mp4"
	@echo "make down / reset stop services / stop and DELETE all volumes"

env:
	@test -f .env || (cp .env.example .env && echo "Created .env - edit the password.")

up: env
	$(COMPOSE) up -d

up-cams: env
	$(COMPOSE) --profile cameras up -d

down:
	$(COMPOSE) --profile cameras down

reset:
	$(COMPOSE) --profile cameras down -v

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

check:
	python scripts/check_infra.py

data:
	python scripts/download_data.py fetch

data-verify:
	python scripts/download_data.py verify

test:
	pytest

test-infra:
	pytest -m infra

SEQ ?= MOT17-04-SDP
mot-video:
	ffmpeg -y -framerate 30 -i data/raw/mot17/MOT17/train/$(SEQ)/img1/%06d.jpg \
	  -c:v libx264 -pix_fmt yuv420p data/sample_videos/sample.mp4
