from huggingface_hub import HfApi, login

login()

repo = "Beshr-Sofi/quote-gpt2"  
api = HfApi()
api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path="models/final", repo_id=repo)