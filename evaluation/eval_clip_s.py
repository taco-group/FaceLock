import argparse
import pandas as pd
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from tqdm import trange
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import edit_prompts, get_image_embeddings, get_text_embeddings

def get_image_embeddings(images, clip_processor, clip_model, device="cuda"):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
        clip_dict = {"method": "clean"}
        clip_scores = []
        seed_dir = os.path.join(clean_edit_dir, f"seed{seed}")
        for i in range(prompt_num):
            prompt_dir = os.path.join(seed_dir, f"prompt{i}")
            cur_prompt = edit_prompts[i]
            text_embeddings = get_text_embeddings(cur_prompt, clip_processor, clip_model, device)
            assert os.path.exists(prompt_dir)
            clip_i = 0
            edit_image_files = sorted(os.listdir(prompt_dir))
            for k in trange(num):
                src_image = Image.open(os.path.join(src_dir, src_image_files[k])).convert("RGB")
                edit_image = Image.open(os.path.join(prompt_dir, edit_image_files[k])).convert("RGB")
                src_embeddings = get_image_embeddings(src_image, clip_processor, clip_model, device)
                edit_embeddings = get_image_embeddings(edit_image, clip_processor, clip_model, device)
                delta_embeddings = edit_embeddings - src_embeddings
                similarity_score = F.cosine_similarity(delta_embeddings, text_embeddings).item()
                clip_i += similarity_score
            clip_i /= num
            clip_scores.append(clip_i)
            clip_dict[f"prompt{i}"] = clip_i
        
        clip_dict["mean"] = np.mean(clip_scores)
        result.append(clip_dict)

        df = pd.DataFrame(result)
        print(df)
        df.to_csv("clip_s_metric.csv", index=False)
    
    for edit_dir in defend_edit_dirs:
        eps_dirs = sorted(os.listdir(edit_dir))
        for eps_dir in eps_dirs:
            cur_method = os.path.join(edit_dir, eps_dir)
            clip_dict = {"method": cur_method}
            clip_scores = []
            print(f"Processing {cur_method}")
            seed_dir = os.path.join(cur_method, f"seed{seed}")
            for i in range(prompt_num):
                prompt_dir = os.path.join(seed_dir, f"prompt{i}")
                cur_prompt = edit_prompts[i]
                text_embeddings = get_text_embeddings(cur_prompt, clip_processor, clip_model, device)
                assert os.path.exists(prompt_dir)
                clip_i = 0
                edit_image_files = sorted(os.listdir(prompt_dir))
                for k in trange(num):
                    src_image = Image.open(os.path.join(src_dir, src_image_files[k])).convert("RGB")
                    edit_image = Image.open(os.path.join(prompt_dir, edit_image_files[k])).convert("RGB")
                    src_embeddings = get_image_embeddings(src_image, clip_processor, clip_model, device)
                    edit_embeddings = get_image_embeddings(edit_image, clip_processor, clip_model, device)
                    delta_embeddings = edit_embeddings - src_embeddings
                    similarity_score = F.cosine_similarity(delta_embeddings, text_embeddings).item()
                    clip_i += similarity_score
                clip_i /= num
                clip_scores.append(clip_i)
                clip_dict[f"prompt{i}"] = clip_i
            
            clip_dict["mean"] = np.mean(clip_scores)
            result.append(clip_dict)

            df = pd.DataFrame(result)
            print(df)
            df.to_csv("clip_s_metric.csv", index=False)
