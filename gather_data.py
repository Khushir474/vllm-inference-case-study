from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='AIxBlock/92k-real-world-call-center-scripts-english', 
                         repo_type='dataset', 
                         local_dir=local_dir)
print('Downloaded to:', path)