import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", 
    type=str, 
    default="PeterJinGo/wiki-18-e5-index", 
    choices=["PeterJinGo/wiki-18-e5-index", "PeterJinGo/wiki-18-corpus"],
    help="Hugging Face repository ID")
parser.add_argument("--save_path", 
    type=str, 
    default="/data/yanfei/SES/local_wiki_server/wiki-18-e5-index", 
    help="Local directory to save files")
    
args = parser.parse_args()

repo_id = "PeterJinGo/wiki-18-e5-index"
for file in ["part_aa", "part_ab"]:
# for file in ["part_ab"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=file,  # e.g., "e5_Flat.index"
        repo_type="dataset",
        local_dir=args.save_path,
    )

repo_id = "PeterJinGo/wiki-18-corpus"
hf_hub_download(
        repo_id=repo_id,
        filename="wiki-18.jsonl.gz",
        repo_type="dataset",
        local_dir=args.save_path,
)