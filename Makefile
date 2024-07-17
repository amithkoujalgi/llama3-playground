build-image:
	docker image prune -f; docker build -t llama3-playground:0.1 .;

build-wheel:
	python build_wheel.py;

run-gpu:
	docker container prune -f; docker run --gpus=all \
	    --shm-size 8G \
		-it \
		-p "8883:8070" \
		-p "8884:9001" \
		-p "8885:8885" \
		-p "8886:8886" \
		-p "8887:8887" \
		-p "8888:8888" \
		-v ~/llama3-playground-data:/app/data \
		llama3-playground:0.1;

run:
	docker container prune -f; docker run \
		-it \
		-p "8883:8070" \
		-p "8884:9001" \
		-p "8885:8885" \
		-p "8886:8886" \
		-p "8887:8887" \
		-p "8888:8888" \
		-v ~/llama3-playground-data:/app/data \
		llama3-playground:0.1;

stop:
	@container_ids=$$(docker ps -a --filter ancestor="llama3-playground:0.1" --format="{{.ID}}"); \
	if [ -n "$$container_ids" ]; then \
		docker stop $$container_ids && docker rm $$container_ids; \
	else \
		echo "No containers to stop."; \
	fi