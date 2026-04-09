# Makefile to orchestrate build and docker-compose operations
SHELL := /bin/bash

.PHONY: help build-images up down rebuild

help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  build-images   - Build frontend and backend Docker images"
	@echo "  up             - Start services (docker compose up -d)"
	@echo "  down           - Stop services (docker compose down)"
	@echo "  rebuild        - Rebuild images and restart services"

build-images:
	chmod +x scripts/build_images.sh
	./scripts/build_images.sh

up:
	# Use system docker compose if available, prefer `docker compose`
	if docker compose version >/dev/null 2>&1; then \
		docker compose -f docker-compose-local.yml up -d; \
	else \
		docker-compose-local -f docker-compose-local.yml up -d; \
	fi

down:
	if docker compose version >/dev/null 2>&1; then \
		docker compose -f docker-compose-local.yml down; \
	else \
		docker-compose-local -f docker-compose-local.yml down; \
	fi

rebuild:
	$(MAKE) build-images
	$(MAKE) down
	echo "Waiting 1s for resources to free..."
	sleep 1
	$(MAKE) up
