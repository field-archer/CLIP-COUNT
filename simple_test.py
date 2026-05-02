import torch
import torch.nn.functional as F
import numpy as np
import argparse
from models import clip_count
from util.FSC147 import FSC147
from util.constant import SCALE_FACTOR
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # 加载模型结构（与训练时保持一致）
    model = clip_count.CLIPCount(
        fim_depth=2,
        fim_num_heads=4,
        use_coop=True,
        use_vpt=True,
        coop_width=2,
        vpt_width=20,
        vpt_depth=10,
        backbone='b16',
        use_fim=False,
        use_mixed_fim=True,
        unfreeze_vit=False
    )
    model = model.to(args.device)
    model.eval()

    # 加载权重（移除前缀 'model.'）
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # 加载数据集
    dataset = FSC147(split=args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    mae = 0.0
    rmse = 0.0
    cnt = 0

    for img, gt_density, boxes, m_flag, prompt in tqdm(dataloader, desc=f"Evaluating {args.split}"):
        img = img.to(args.device)
        gt = gt_density.to(args.device)
        # prompt 是 tuple of string，需要提取
        prompt_text = prompt[0]  # shape (1,)
        with torch.no_grad():
            output = model(img, [prompt_text])  # 输出密度图 (1, H, W)

        pred_cnt = torch.sum(output / SCALE_FACTOR).item()
        gt_cnt = torch.sum(gt / SCALE_FACTOR).item()
        err = abs(pred_cnt - gt_cnt)
        mae += err
        rmse += err ** 2
        cnt += 1

    mae /= cnt
    rmse = (rmse / cnt) ** 0.5
    print(f"Split: {args.split}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == '__main__':
    main()