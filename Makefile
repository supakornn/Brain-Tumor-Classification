HF_REPO = supakornn/Brain-Tumor-Classification

.PHONY: train visualize push-github push-hf publish

train:
	uv run python scripts/train_models.py

visualize:
	uv run python scripts/data_exploration.py

push-github:
	git add -A
	git commit -m "chore: update" || true
	git push origin main

push-hf:
	huggingface-cli upload $(HF_REPO) hf/README.md README.md --repo-type model
	huggingface-cli upload $(HF_REPO) images/ images/ --repo-type model
	huggingface-cli upload $(HF_REPO) model/ model/ --repo-type model

publish: push-github push-hf
