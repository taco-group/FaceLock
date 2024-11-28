import argparse
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import trange
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import edit_prompts, load_model_by_repo_id, compute_score, pil_to_input

repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
aligner_id = 'minchul/cvlface_DFA_mobilenet'
# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fr_model = load_model_by_repo_id(repo_id=repo_id,
                                     save_path=f'{os.environ["HF_HOME"]}/{repo_id}',
                                     HF_TOKEN=os.environ['HUGGINGFACE_HUB_TOKEN']).to(device)
aligner = load_model_by_repo_id(repo_id=aligner_id,
                                    save_path=f'{os.environ["HF_HOME"]}/{aligner_id}',
                                    HF_TOKEN=os.environ['HUGGINGFACE_HUB_TOKEN']).to(device)

if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, type=str, help="path to the directory containing the source images")
    parser.add_argument("--clean_edit_dir", default=None, help="path to the directory containing edits on the unprotected images")
    parser.add_argument("--defend_edit_dirs", required=True, nargs="+", help="path to the directory containing different attack budget subdirectories")
    parser.add_argument("--seed", required=True, type=int, help="the seed to evaluate on")
    args = parser.parse_args()

    src_dir = args.src_dir
    src_image_files = sorted(os.listdir(src_dir))
    num = len(src_image_files)
    defend_edit_dirs = args.defend_edit_dirs
    prompt_num = len(edit_prompts)
    seed = args.seed
    for x in defend_edit_dirs:
        assert os.path.exists(x)

    result = []

    if args.clean_edit_dir is not None:
        clean_edit_dir = args.clean_edit_dir
        print("Processing clean")
        facial_dict = {"method": "clean"}
        facial_scores = []
        seed_dir = os.path.join(clean_edit_dir, f"seed{seed}")
        for i in range(prompt_num):
            prompt_dir = os.path.join(seed_dir, f"prompt{i}")
            assert os.path.exists(prompt_dir)
            facial_i = 0
            edit_image_files = sorted(os.listdir(prompt_dir))
            for k in trange(num):
                src_image = Image.open(os.path.join(src_dir, src_image_files[k])).convert("RGB")
                edit_image = Image.open(os.path.join(prompt_dir, edit_image_files[k])).convert("RGB")
                similarity_score = compute_score(pil_to_input(src_image).cuda(), pil_to_input(edit_image).cuda(), aligner, fr_model).item()
                facial_i += similarity_score
            facial_i /= num
            facial_scores.append(facial_i)
            facial_dict[f"prompt{i}"] = facial_i
        
        facial_dict["mean"] = np.mean(facial_scores)
        result.append(facial_dict)

        df = pd.DataFrame(result)
        print(df)
        df.to_csv("facial_metric.csv", index=False)
    
    for edit_dir in defend_edit_dirs:
        eps_dirs = sorted(os.listdir(edit_dir))
        for eps_dir in eps_dirs:
            cur_method = os.path.join(edit_dir, eps_dir)
            facial_dict = {"method": cur_method}
            facial_scores = []
            print(f"Processing {cur_method}")
            seed_dir = os.path.join(cur_method, f"seed{seed}")
            for i in range(prompt_num):
                prompt_dir = os.path.join(seed_dir, f"prompt{i}")
                assert os.path.exists(prompt_dir)
                facial_i = 0
                edit_image_files = sorted(os.listdir(prompt_dir))
                for k in trange(num):
                    src_image = Image.open(os.path.join(src_dir, src_image_files[k])).convert("RGB")
                    edit_image = Image.open(os.path.join(prompt_dir, edit_image_files[k])).convert("RGB")
                    similarity_score = compute_score(pil_to_input(src_image).cuda(), pil_to_input(edit_image).cuda(), aligner, fr_model).item()
                    facial_i += similarity_score
                facial_i /= num
                facial_scores.append(facial_i)
                facial_dict[f"prompt{i}"] = facial_i
            
            facial_dict["mean"] = np.mean(facial_scores)
            result.append(facial_dict)

            df = pd.DataFrame(result)
            print(df)
            df.to_csv("facial_metric.csv", index=False)
