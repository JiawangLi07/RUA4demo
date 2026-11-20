import sys
import torch
import numpy as np
import argparse
from inference import Inference, load_image, load_single_mask
from fft.fft2d import calculate_hfer_robust
import os
import time
from omegaconf import OmegaConf, DictConfig, ListConfig

sys.path.append("notebook")

def clear_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            return False
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f"Error {file_path} : {e}")
        return True
        
    except Exception as e:
        return False

def inspect_dict(output_dict):
  
    for key, value in output_dict.items():
        if isinstance(value, torch.Tensor):
            info = str(list(value.shape))
            type_name = "torch.Tensor"
        elif isinstance(value, np.ndarray):
            info = str(value.shape)
            type_name = "np.ndarray"
        elif isinstance(value, list):
            info = f"len={len(value)}"
            type_name = "List"
        else:
            info = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
            type_name = type(value).__name__          

def save_visual_ply(gs_model, path):
    from plyfile import PlyData, PlyElement
    folder_path = os.path.dirname(path)
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    xyz = gs_model._xyz.detach().cpu().numpy()
    f_dc = gs_model._features_dc.detach().contiguous().cpu().numpy()
    SH_C0 = 0.28209479177387814
    rgb = 0.5 + (SH_C0 * f_dc)
    
    rgb = np.clip(rgb, 0, 1) * 255
    rgb = rgb.astype(np.uint8)
    rgb = rgb.squeeze(1)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def main():
    parser = argparse.ArgumentParser(description="3D GS Inference Script")
    
    parser.add_argument("--tag", type=str, default="hf", help="model Tag")
    parser.add_argument("--image_path", type=str, required=True, help="image path")
    parser.add_argument("--mask_index", type=int, default=14, help="mask index")
    parser.add_argument("--output_dir", type=str, default="./Generate", help="ply and glb")
    
    parser.add_argument("--seed", type=int, default=42, help="seed")
    #---SSG
    parser.add_argument("--ss_faster_stride", type=int, default=3)
    parser.add_argument("--ss_warmup", type=int, default=2)
    parser.add_argument("--ss_order", type=int, default=1)
    parser.add_argument("--ss_momentum_beta", type=float, default=0.5)
    #---SLaT
    parser.add_argument("--slat_thresh", type=float, default=0.5)
    parser.add_argument("--slat_warmup", type=int, default=2)
    parser.add_argument("--slat_token_ratio", type=float, default=0.15)
    #--Mesh
    parser.add_argument("--mesh_spectral_threshold_low", type=float, default=0.5)
    parser.add_argument("--mesh_spectral_threshold_high", type=float, default=0.7)
    #--Open
    parser.add_argument("--enable_ss_faster", action="store_true", help="")
    parser.add_argument("--enable_slat_token", action="store_true", help="")
    parser.add_argument("--enable_mesh_aggregation", action="store_true", help="")
    parser.add_argument("--enable_acceleration", action="store_true", help="")

    # --more
    parser.add_argument("--enable_taylor", action="store_true", help="")
    parser.add_argument("--enable_easy", action="store_true", help="")
    parser.add_argument("--steps", type=int, help="Denoising steps per stage")
    parser.add_argument("--voxel-only", action="store_true",
                        help="Stop after stage 1 and export occupied grid cells as NPZ")
    parser.add_argument("--distill-stage1", action="store_true",
                        help="Enable stage-1 shortcut distillation inference")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")

    
    def get_enable_params(args):
        args_dict = vars(args)
        enable_params = {k: v for k, v in args_dict.items() if k.startswith("enable_")}
        
        if enable_params.get('enable_acceleration', False):
            enable_params['enable_ss_faster'] = True
            enable_params['enable_slat_token'] = True
            enable_params['enable_mesh_aggregation'] = True

        if enable_params.get('enable_taylor', False) or enable_params.get('enable_easy', False):
            enable_params['enable_ss_faster'] = False
            enable_params['enable_slat_token'] = False
            enable_params['enable_mesh_aggregation'] = False
        
        for k, v in enable_params.items():
            setattr(args, k, v)
        
        return enable_params

    config_path = f"checkpoints/{args.tag}/pipeline.yaml"
    enable_params = get_enable_params(args)
    config = OmegaConf.load(config_path) 
    config.workspace_dir = os.path.dirname(config_path)
    if args.steps is not None:
        config.ss_inference_steps = args.steps
        config.slat_inference_steps = args.steps

    if enable_params['enable_ss_faster']:
        config['ss_generator_config_path'] =  "ss_generator_faster.yaml" 
    if enable_params['enable_slat_token']:
        config['slat_generator_config_path'] = "slat_generator_faster.yaml" 

    # This option has higher priority.
    if enable_params['enable_taylor']:
        config['ss_generator_config_path'] =  "ss_generator_taylorseer.yaml"
        config['slat_generator_config_path'] = "slat_generator_taylorseer.yaml" 

    if enable_params['enable_easy']:
        config['ss_generator_config_path'] =  "ss_generator_easy.yaml"
        config['slat_generator_config_path'] = "slat_generator_easy.yaml"

    print(f"Acceleration enabled: SS:{enable_params['enable_ss_faster']}, SLaT:{enable_params['enable_slat_token']}, Mesh:{enable_params['enable_mesh_aggregation']}")
    
    inference = Inference(config, compile=False, args=args)


    # load image and mask
    image = load_image(args.image_path)
    folder_path = os.path.dirname(args.image_path)
    mask_path = os.path.join(folder_path, f"{args.mask_index}.png")
    mask = load_single_mask(folder_path, index=args.mask_index)
    
    hfer = calculate_hfer_robust(mask_path)

    if hasattr(inference, 'get_hfer'):
        inference.get_hfer(hfer)
    if hasattr(inference, 'get_params'):
        inference.get_params(args)

    # inferece
    print(f"Start inference: {args.image_path}")
    torch.cuda.synchronize()
    s_time = time.perf_counter()
    output = inference(
        image, 
        mask, 
        seed=args.seed,
        stage1_only=args.voxel_only,
        use_stage1_distillation=args.distill_stage1,
    )
    torch.cuda.synchronize()
    print(f"Done, total time: {time.perf_counter() - s_time:.2f}s")


    os.makedirs(args.output_dir, exist_ok=True)
    if args.voxel_only:
        path = os.path.join(args.output_dir, f"voxels-{args.mask_index}.npz")
        np.savez_compressed(
            path,
            native_coords=output["coords_original"][:, 1:].cpu().numpy(),
            coords=output["coords"][:, 1:].cpu().numpy(),
            voxel=output["voxel"].cpu().numpy(),
            downsample_factor=float(output["downsample_factor"]),
        )
        print(f"Saved occupied cells to {path}")
        return
    ply_path = os.path.join(args.output_dir, f"splat-faster-{args.mask_index}.ply")
    save_visual_ply(output["gs"], ply_path)
    glb_path = os.path.join(args.output_dir, f"splat-faster-{args.mask_index}.glb")
    output["glb"].export(glb_path)
    print(f"Files saved to:\n - {ply_path}\n - {glb_path}")

if __name__ == "__main__":
    main()
