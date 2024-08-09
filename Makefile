image:
	docker image prune -f; docker build -t amithkoujalgi/llama3-playground:0.1 .;

push:
	docker push amithkoujalgi/llama3-playground:0.1;

wheel:
	python build_wheel.py;

start-gpu:
	docker container prune -f; docker run --gpus=all \
	    --shm-size 8G \
		-itd \
		-p "8883:8070" \
		-p "8884:9001" \
		-p "8885:8885" \
		-p "8886:8886" \
		-p "8887:8887" \
		-p "8888:8888" \
		-v ~/llama3-playground-data:/app/data \
		amithkoujalgi/llama3-playground:0.1;

start:
	docker container prune -f; docker run \
		-itd \
		-p "8883:8070" \
		-p "8884:9001" \
		-p "8885:8885" \
		-p "8886:8886" \
		-p "8887:8887" \
		-p "8888:8888" \
		-v ~/llama3-playground-data:/app/data \
		amithkoujalgi/llama3-playground:0.1;

stop:
	@container_ids=$$(docker ps -a --filter ancestor="amithkoujalgi/llama3-playground:0.1" --format="{{.ID}}"); \
	if [ -n "$$container_ids" ]; then \
		docker stop $$container_ids && docker rm $$container_ids; \
	else \
		echo "No containers to stop."; \
	fi