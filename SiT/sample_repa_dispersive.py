# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time
import torch.nn.functional as F
from PIL import Image

def custom_l2_loss(features: torch.Tensor, temperature: float = 1,metric:str='l2') -> torch.Tensor:

    # features : (batch_size, seq_len, feature_dim)

    if metric == 'l2':
        N = features.shape[0]
        features_flat = features.reshape(N, -1)  # (N, D)

        # features_flat = features_flat / features_flat.norm(p=2, dim=1, keepdim=True).max()
        # diff = features_flat.unsqueeze(1) - features_flat.unsqueeze(0)  # (N, N, D)
        # torch.cdist를 사용하여 (N, N, D) 크기의 중간 텐서 생성을 피해 메모리 사용량을 최적화합니다.
        D = torch.cdist(features_flat, features_flat, p=2).pow(2)  # (N, N)
        # breakpoint()
        mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        D = D[mask]
        # breakpoint()
        loss  = D.sum()
        # breakpoint()
        return loss

    else:
        raise ValueError(f"Invalid metric: {metric}")

def info_nce_loss(features: torch.Tensor, temperature: float = 0.1, metric:str='l2') -> torch.Tensor:

    # features : (batch_size, seq_len, feature_dim)

    if metric == 'l2':
        N = features.shape[0]
        features_flat = features.reshape(N, -1)  # (N, D)

        features_flat = features_flat / features_flat.norm(p=2, dim=1, keepdim=True).max()
        # diff = features_flat.unsqueeze(1) - features_flat.unsqueeze(0)  # (N, N, D)
        # torch.cdist를 사용하여 (N, N, D) 크기의 중간 텐서 생성을 피해 메모리 사용량을 최적화합니다.
        D = torch.cdist(features_flat, features_flat, p=2).pow(2)  # (N, N)
        # breakpoint()
        mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        D = D[mask]
        # breakpoint()
        max_D = torch.max(-D / temperature)
        loss = max_D + torch.log(torch.mean(torch.exp(-D / temperature - max_D)))
        # breakpoint()
        return loss
    elif metric=='cosine':
        N=features.shape[0]
        pred_flat = features.reshape(N, -1)
        pred_norm = F.normalize(pred_flat, p=2, dim=1)
        sim_matrix = torch.matmul(pred_norm, pred_norm.t())
        mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        D = 1 - sim_matrix.masked_select(mask)
        return torch.log(torch.mean(torch.exp(-D/temperature)))
    else:
        raise ValueError(f"Invalid metric: {metric}")

@torch.enable_grad()
def hook_fn(module, input, output):

    default_dtype = output.dtype
    upcast_dtype = torch.float32
    features = output.detach().to(upcast_dtype) # (batch_size, seq_len, feature_dim)



    # CFG 
    # target_feature = other_features_mean + (target_feature - other_features_mean) * args.update_scale

    # Gradient Ascent Ver1
    # loss = F.mse_loss(target_feature, other_features, reduction='sum')
    # grad = torch.autograd.grad(loss, target_feature)[0] / len(features) # 평균을 내기 위해 배치 크기로 나눔
    # if grad is None:
    #     raise ValueError("Gradient가 None입니다")
    # target_feature = target_feature + grad * args.update_scale

    # Custom L2 Loss
    if args.loss == 'CustomL2':
        target_feature = features[0].detach().requires_grad_(True) # Gradient를 계산하기 위해 requires_grad=True로 설정
        other_features = features[1:] # 나머지 특징들
        other_features_mean = features[1:].mean(dim=0)

        print(f"CustomL2 Loss를 사용합니다. loss_style: {args.loss_style}")
        new_features = torch.cat([target_feature.unsqueeze(0), other_features], dim=0) # (batch_size, seq_len, feature_dim)
        loss = custom_l2_loss(new_features, temperature=0.1, metric=args.loss_style) # InfoNCE 손실 계산
        grad = torch.autograd.grad(loss, target_feature)[0]
        if grad is None:
            raise ValueError("Gradient가 None입니다")
        target_feature = target_feature + torch.norm(target_feature)/torch.norm(grad)*0.1 * grad * args.update_scale # Gradient를 업데이트

        features = torch.cat([target_feature.unsqueeze(0), other_features], dim=0)


    elif args.loss == "GradientAscent":
        target_feature = features[0].detach().requires_grad_(True) # Gradient를 계산하기 위해 requires_grad=True로 설정
        other_features = features[1:] # 나머지 특징들
        other_features_mean = features[1:].mean(dim=0)

        loss = F.mse_loss(target_feature, other_features, reduction='sum')
        grad = torch.autograd.grad(loss, target_feature)[0]
        if grad is None:
            raise ValueError("Gradient가 None입니다")
        target_feature = target_feature + torch.norm(target_feature)/torch.norm(grad)*0.1 * grad * args.update_scale

        features = torch.cat([target_feature.unsqueeze(0), other_features], dim=0)


    # InfoNCE Loss
    elif args.loss == 'InfoNCE':
        target_feature = features[0].detach().requires_grad_(True) # Gradient를 계산하기 위해 requires_grad=True로 설정
        other_features = features[1:] # 나머지 특징들
        other_features_mean = features[1:].mean(dim=0)

        new_features = torch.cat([target_feature.unsqueeze(0), other_features], dim=0) # (batch_size, seq_len, feature_dim)
        loss = info_nce_loss(new_features, temperature=0.1, metric=args.loss_style) # InfoNCE 손실 계산
        grad = torch.autograd.grad(loss, target_feature)[0]
        if grad is None:
            raise ValueError("Gradient가 None입니다")
        target_feature = target_feature - torch.norm(target_feature)/torch.norm(grad)*0.1 * grad * args.update_scale # Gradient를 업데이트

        features = torch.cat([target_feature.unsqueeze(0), other_features], dim=0)

    # InfoNCE Loss
    elif args.loss == 'InfoNCE_entire':
        new_features = features.requires_grad_(True) # 전체 특징을 업데이트하기 위해 requires_grad=True로 설정

        loss = info_nce_loss(new_features, temperature=0.1, metric=args.loss_style)

        grads = torch.autograd.grad(loss, new_features, retain_graph=False)[0]  # (batch_size, seq_len, feature_dim)

        if grads is None:
            raise ValueError("Gradient가 None입니다")

        # Gradient normalization factor (optional, same scale as others)
        norm_factor = torch.norm(new_features, dim=(1,2), keepdim=True) / (torch.norm(grads, dim=(1,2), keepdim=True) + 1e-8)
        updates = -norm_factor * 0.1 * grads * args.update_scale

        features = new_features + updates
    
    else :
        raise ValueError(f"지원하지 않는 손실 함수: {args.loss}. 'InfoNCE' 또는 'CustomL2'만 지원합니다.")

    


    # Gradient Ascent Ver2 (Same w/ CFG)
    # loss = F.mse_loss(target_feature, other_features_mean, reduction='sum')
    # grad = torch.autograd.grad(loss, target_feature)[0]
    # if grad is None:
    #     raise ValueError("Gradient가 None입니다")
    # target_feature = target_feature + grad * args.update_scale / 2 # 2로 나눠주면 CFG하고 완벽히 똑같은 식임

    features = features.to(default_dtype)
    new_output = features

    return new_output


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False

    learn_sigma = False # REPA는 sigma를 학습하지 않음

    # Load model:
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or "/media/dataset2/jiwon/representation/SiT/checkpoint/repa.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)


    # Hook the model
    layers_to_hook = [args.layer] if not args.baseline else []  # 특정 레이어만 훅을 걸거나, baseline 모드에서는 훅을 사용하지 않음
    hook_handle_list = []
    for layer_idx in layers_to_hook:
        hook_handle = model.blocks[layer_idx].register_forward_hook(hook_fn)
        hook_handle_list.append(hook_handle)
    dispersive_kwargs = {
        'hook_handle_list': hook_handle_list,
        'using_steps': args.using_steps,  # 몇 번째 디노이징 스텝에서 훅을 해제할지
    }


    # Labels to condition the model with (feel free to change):
    import random
    import numpy as np
    class_labels = np.linspace(0, args.num_classes - 1, args.batch_size).astype(int).tolist() # 배치 사이즈만큼 균등 샘플링
    
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    z = z.repeat(n, 1, 1, 1)  # 동일 Noise
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    using_cfg = args.cfg_scale > 1.0

    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_fn = model.forward

    # Sample images:
    start_time = time()
    samples = sample_fn(z, model_fn, dispersive_kwargs, **model_kwargs)[-1]
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    save_path = os.path.join(args.save_path, 'repa', args.exp_name)
    if len(layers_to_hook) > 0: 
        save_path = os.path.join(save_path, f"seed{args.seed}_batch_{args.batch_size}_layer_{layers_to_hook}_scale_{args.update_scale}_using_steps_{args.using_steps}")
    else:
        save_path = os.path.join(save_path, f"baseline_cfg{args.cfg_scale}")

    # Save and display images:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    from tqdm import tqdm
    for i, sample in tqdm(enumerate(samples), total=len(samples), desc="Saving images"):
        sample = (sample + 1) / 2  # Normalize to [0, 1]
        ndarr = sample.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(os.path.join(save_path, f"img_{i}.png"))

    save_image(samples, os.path.join(save_path, "grid.png"), nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--num-sampling-steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    # Dispersive
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기. InfoNCE 손실을 위해 1보다 커야 합니다.")
    parser.add_argument("--loss", type=str, default='InfoNCE_entire', choices=['InfoNCE', 'InfoNCE_entire', 'CustomL2', 'GradientAscent'], help="사용할 손실 함수")
    parser.add_argument("--loss_style", type=str, default='l2', choices=['l2', 'cosine'], help="InfoNCE 손실을 위한 메트릭")
    parser.add_argument("--using_steps", type=int, default=9, help="특징을 추출할 디노이징 스텝")
    parser.add_argument("--save_path", type=str, default="./outputs_jiwon", help="save path")
    parser.add_argument("--exp_name", type=str, default="InfoNCE_NormScale_entire", help="experiment name")
    parser.add_argument("--update_scale", type=float, default=1.0, help="update scale")
    parser.add_argument("--layer", type=int, default=1, help="layer to hook (0-27 for SiT-XL/2)")
    parser.add_argument("--baseline", action="store_true", help="baseline mode, no dispersive sampling")


    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
