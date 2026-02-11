# runai_tutorials
megatron + nccl

runai secret create nvcr-creds \
  -p nccl-benchmarking \
  --type=docker-registry \
  --server=nvcr.io \
  --username='$oauthtoken' \
  --password='<YOUR_NVCR_API_KEY>'