from modelscope import snapshot_download

model_dir = "models/Qwen3-1.7B"
model_dir = snapshot_download('Qwen/Qwen3-1.7B', local_dir=model_dir)
print(model_dir)