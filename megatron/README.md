export NGC_API_KEY='EXAMPLE_KEY'

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin


export CONTAINER_ID=riva-translate-1_6b

docker run -it --rm --name=$CONTAINER_ID \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_HTTP_API_PORT=9000 \
  -e NIM_GRPC_API_PORT=50051 \
  -p 9000:9000 \
  -p 50051:50051 \
  nvcr.io/nim/nvidia/$CONTAINER_ID:latest


curl -X 'GET' 'http://localhost:9000/v1/health/ready'

sudo apt-get install python3-pip
pip install -U nvidia-riva-client

git clone https://github.com/nvidia-riva/python-clients.git

python3 python-clients/scripts/nmt/nmt.py --server 0.0.0.0:50051 \
    --text "This will become German words" \
    --source-language-code en-US \
    --target-language-code de-DE