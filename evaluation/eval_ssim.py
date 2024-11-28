# Compute ssim between the edits on protected and unprotected images
import argparse
import pandas as pd
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import trange
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import edit_prompts

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_edit_dir", required=True, type=str, help="path to the directory containing edits on the unprotected images")
    parser.add_argument("--defend_edit_dirs", required=True, nargs="+", help="path to the directory containing different attack budget subdirectories")
    parser.add_argument("--seed", required=True, type=int, help="the seed to evaluate on")
    args = parser.parse_args()

    clean_edit_dir = args.clean_edit_dir
    defend_edit_dirs = args.defend_edit_dirs
    prompt_num = len(edit_prompts)
    seed = args.seed
    for x in defend_edit_dirs:
        assert os.path.exists(x)

    result = []
    
    for edit_dir in defend_edit_dirs:
        eps_dirs = sorted(os.listdir(edit_dir))
        for eps_dir in eps_dirs:
            cur_method = os.path.join(edit_dir, eps_dir)
            ssim_dict = {"method": cur_method}
            ssim_scores = []
            print(f"Processing {cur_method}")
            seed_dir = os.path.join(cur_method, f"seed{seed}")
            clean_seed_dir = os.path.join(clean_edit_dir, f"seed{seed}")
            for i in range(prompt_num):
                prompt_dir = os.path.join(seed_dir, f"prompt{i}")
                clean_prompt_dir = os.path.join(clean_seed_dir, f"prompt{i}")
                assert os.path.exists(prompt_dir) and os.path.exists(clean_prompt_dir)
                ssim_i = 0
                image_files = sorted(os.listdir(prompt_dir))
                clean_image_files = sorted(os.listdir(clean_prompt_dir))
                num = len(image_files)
                assert num == len(clean_image_files)
                for k in trange(num):
                    edit_image = Image.open(os.path.join(prompt_dir, image_files[k])).convert("RGB")
                    clean_edit_image = Image.open(os.path.join(clean_prompt_dir, clean_image_files[k])).convert("RGB")
                    edit_tensor = T.ToTensor()(edit_image).unsqueeze(0)
                    clean_edit_tensor = T.ToTensor()(clean_edit_image).unsqueeze(0)
                    ssim_i += ssim(edit_tensor, clean_edit_tensor).item()
                ssim_i /= num
                ssim_scores.append(ssim_i)
                ssim_dict[f"prompt{i}"] = ssim_i
            
            ssim_dict["mean"] = np.mean(ssim_scores)
            result.append(ssim_dict)

            df = pd.DataFrame(result)
            print(df)
            df.to_csv("ssim_metric.csv", index=False)
