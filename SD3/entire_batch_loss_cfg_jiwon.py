import os
import argparse
import torch
from diffusers import StableDiffusion3Pipeline
# loss.py가 동일한 'representation' 디렉토리에 있다고 가정합니다.
from loss import info_nce_loss
from typing import List
import torch.nn.functional as F


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

def info_nce_loss(features: torch.Tensor, temperature: float = 1,metric:str='l2') -> torch.Tensor:

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

def main():
    # --- 인수 파싱 (Argument Parsing) ---
    parser = argparse.ArgumentParser(description="representation training")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers", help="Hugging Face Hub의 모델 ID")
    # parser.add_argument("--prompt", type=str, default="A photo of a dog", help="이미지 생성을 위한 프롬프트")
    parser.add_argument("--layer_index", type=int,default=1, help="특징을 추출할 트랜스포머 블록의 인덱스")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Classifier-Free Guidance 스케일")
    parser.add_argument("--batch_size", type=int, default=7, help="배치 크기. InfoNCE 손실을 위해 1보다 커야 합니다.")
    parser.add_argument("--loss", type=str, default='InfoNCE', choices=['InfoNCE', 'InfoNCE_entire', 'CustomL2', 'GradientAscent'], help="사용할 손실 함수")
    parser.add_argument("--loss_style", type=str, default='l2', choices=['l2', 'cosine'], help="InfoNCE 손실을 위한 메트릭")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="디노이징 스텝 수")
    parser.add_argument("--seed", type=int, default=10, help="재현성을 위한 랜덤 시드")
    parser.add_argument("--using_steps", type=int, default=9, help="특징을 추출할 디노이징 스텝")
    parser.add_argument("--batch_option", type=bool, default="diff_noise", help="latent 특징을 추출할지 여부")
    parser.add_argument("--update_method", type=str, default="direct", choices=["reverse", "direct"], help="learning rate")
    parser.add_argument("--save_path", type=str, default="./outputs_jiwon", help="save path")
    parser.add_argument("--exp_name", type=str, default="InfoNCE", help="experiment name")
    parser.add_argument("--update_scale", type=float, default=2, help="update scale")
    parser.add_argument("--batch_loss", type=bool, default=False, help="Apply loss dynamically on the current batch during inference (no pre-computed features)")
    parser.add_argument("--latent_reinit", type=bool, default=False, help="Reinitialize the latent at every timestep (useful for dynamic loss)")
    args = parser.parse_args()



    if args.batch_size <= 1:
        raise ValueError("InfoNCE 손실을 계산하려면 배치 크기가 1보다 커야 합니다.")

    # --- 설정 (Setup) ---
    prompt_list = [
        # sanity check
        "a photo of a dog",
        # "Iron Man, (Arnold Tsang, Toru Nakayama), Masterpiece, Studio Quality, 6k , toa, toaair, 1boy, glowing, axe, mecha, science_fiction, solo, weapon, jungle , green_background, nature, outdoors, solo, tree, weapon, mask, dynamic lighting, detailed shading, digital texture painting",
        "a photo of a cat", 
        "a photo of a elephant",
        "a photo of a tiger",
        "a photo of a lion",
        "a photo of a giraffe",
        "a photo of a zebra",
        "a photo of a penguin",
        "a photo of a kangaroo",
        "a photo of a panda",
        'a photo of a cow',
        "a photo of a horse",
        "a photo of a fox",
        "a photo of a wolf",
        "a photo of a rabbit",
        "a photo of a bear",
        "a photo of a cheetah",
        "a photo of a leopard",
        "a photo of a buffalo",
        "a photo of a moose",
        "a photo of a deer",
        "a photo of a camel",
        "a photo of a crocodile",
        "a photo of an alligator",
        "a photo of a hippopotamus",
        "a photo of a rhinoceros",
        "a photo of a sloth",
        "a photo of a monkey",
        "a photo of a chimpanzee",
        "a photo of an orangutan",
        "a photo of a baboon",
        "a photo of a parrot",
        "a photo of a flamingo",
        "a photo of a toucan",
        "a photo of a peacock",
        "a photo of an ostrich",
        "a photo of a dove",
        "a photo of an eagle",
        "a photo of a hawk",
        "a photo of a falcon",
        "a photo of an owl",
        "a photo of a bat",
        "a photo of a koala",
        "a photo of a raccoon",
        "a photo of a skunk",
        "a photo of a hedgehog",
        "a photo of a porcupine",
        "a photo of a squirrel",
        "a photo of a chipmunk",
        "a photo of a mole",
        "a photo of a hamster",
        "a photo of a guinea pig",
        "a photo of a mouse",
        "a photo of a rat",
        "a photo of a sea lion",
        "a photo of a seal",
        "a photo of a dolphin",
        "a photo of a whale",
        "a photo of a shark",
        "a photo of a stingray",
        "a photo of a jellyfish",
        "a photo of a starfish",
        "a photo of a turtle",
        "a photo of a frog"
    ]
    args.save_path = os.path.join(args.save_path, prompt_list[0].replace(" ", "_")[:20]) # 첫 번째 프롬프트로 저장 경로 설정
    args.save_path = os.path.join(args.save_path, args.exp_name)
    args.save_path = os.path.join(args.save_path, f"seed{args.seed}_batch_{args.batch_size}_layer_{args.layer_index}_scale_{args.update_scale}_using_steps_{args.using_steps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_device = "cpu"
    # seed_device = device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float16

    # if args.seed is not None:
    #     torch.manual_seed(args.seed)

    # --- 파이프라인 로드 (Load Pipeline) ---
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()

    
    # 모든 모델 컴포넌트를 float32로 통일
    # pipe.text_encoder.to(dtype=torch_dtype)
    # pipe.text_encoder_2.to(dtype=torch_dtype)
    # pipe.text_encoder_3.to(dtype=torch_dtype)
    # pipe.transformer.to(dtype=torch_dtype)
    # pipe.vae.to(dtype=torch_dtype)
    
    # breakpoint()

    # --- 특징 추출 훅 (Feature Extraction Hook) ---
    # 이 리스트는 훅이 걸린 레이어의 특징을 저장합니다.
    extracted_features = []
    step_counter = 0

    @torch.enable_grad()
    def hook_fn(module, input, output):

        nonlocal step_counter
        nonlocal args
        step_counter += 1
        # if step_counter == args.using_steps:

        default_dtype = output[1].dtype
        upcast_dtype = torch.float32
        features = output[1].detach().to(upcast_dtype) # (batch_size, seq_len, feature_dim)



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
        new_output = (output[0], features)

        return new_output


    num_blocks = len(pipe.transformer.transformer_blocks)
    if not 0 <= args.layer_index < num_blocks:
        raise ValueError(f"잘못된 layer_index입니다. 0과 {num_blocks - 1} 사이여야 합니다.")



    # ------------------------------------------------------------------
    # Fast path: directly perform a single batched inference run where the
    # dynamic Info-NCE loss is applied at each timestep. This leverages
    # the `batch_loss` branch in `transformer_sd3.py` and avoids the costly
    # per-prompt feature pre-computation.
    # ------------------------------------------------------------------
    prompts = prompt_list[: args.batch_size]
    if len(prompts) < args.batch_size: # 프롬프트가 부족할 경우 사용자 프롬프트로 채움
        raise ValueError(f"프롬프트의 개수가 배치 크기({args.batch_size})보다 작습니다. {len(prompts)}개의 프롬프트가 필요합니다.")
    
    gen_list = [torch.Generator(seed_device).manual_seed(args.seed) for _ in range(args.batch_size)]

    if args.exp_name == 'baseline':
        results = pipe(
            prompt=prompts,
            num_images_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=gen_list,
            #
            # hook_handle=hook_handle,
            # remove_hook_handle_after=args.using_steps,
            # latent_reinit_every_timestep=True,
            #
            # dispersive_guide=False,
            # dispersive_layer=args.layer_index,
            # loss_style=args.loss_style,
            # using_steps=args.using_steps,
            # update_method=args.update_method,
            # batch_loss=args.batch_loss,
            # update_scale=args.update_scale,
        )

    
    else :
        hook_handle = pipe.transformer.transformer_blocks[args.layer_index].register_forward_hook(hook_fn)
        print(f"latent reinitialization every timestep: {args.latent_reinit}")
        results = pipe(
            prompt=prompts,
            num_images_per_prompt=1,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            #
            generator=gen_list,
            hook_handle=hook_handle,
            remove_hook_handle_after=args.using_steps,
            latent_reinit_every_timestep=args.latent_reinit,
            #
            # dispersive_guide=False,
            # dispersive_layer=args.layer_index,
            # loss_style=args.loss_style,
            # using_steps=args.using_steps,
            # update_method=args.update_method,
            # batch_loss=args.batch_loss,
            # update_scale=args.update_scale,
        )

    os.makedirs(args.save_path, exist_ok=True)
    for idx, img in enumerate(results.images):
        img.save(os.path.join(args.save_path, f"img_{idx}.png"))
        print("이미지가 저장되었습니다:", args.save_path)
    
if __name__ == "__main__":
    main()