"""Benchmark independent masks on one captured RGB image, with a resident model."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
from infer import Inference, OmegaConf, calculate_hfer_robust, load_image

def run():
    parser = ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--compare-distillation', action='store_true')
    options = parser.parse_args()
    capture = options.capture
    config = OmegaConf.load('checkpoints/hf/pipeline.yaml')
    config.workspace_dir = 'checkpoints/hf'
    config.ss_generator_config_path = 'ss_generator_faster.yaml'
    config.slat_generator_config_path = 'slat_generator_faster.yaml'
    config.ss_inference_steps = config.slat_inference_steps = 18
    args = Namespace(ss_faster_stride=3, ss_warmup=2, ss_order=1, ss_momentum_beta=.5,
                     slat_thresh=1.5, slat_warmup=3, slat_token_ratio=.1,
                     mesh_spectral_threshold_low=.5, mesh_spectral_threshold_high=.7,
                     enable_mesh_aggregation=True)
    model = Inference(config, compile=False, args=args)
    model.get_params(args)
    pipe = model._pipeline
    image = load_image(str(capture/'main.png'))
    inputs = []
    for item in json.loads((capture/'instances.json').read_text())['main']:
        name = item['id']
        path = capture/'masks'/'main'/(name+'.png')
        inputs.append((name, model.merge_mask_to_rgba(image,load_image(str(path))>0),
                       calculate_hfer_robust(str(path))))
    def reconstruct(rgba, hfer, steps, shared, depth, distilled=False):
        model.get_hfer(hfer)
        return pipe.run(rgba, seed=42, stage1_only=True, stage1_inference_steps=steps,
                        frame_depth=depth, draw_frequency_plot=not shared,
                        use_stage1_distillation=distilled)
    reconstruct(*inputs[0][1:],18,False,None)
    reference = {}
    report = []
    if options.compare_distillation:
        modes = [('off18',18,True,False), ('on18',18,True,True),
                 ('off12',12,True,False), ('on12',12,True,True),
                 ('on18_repeat',18,True,True), ('off18_repeat',18,True,False)]
        filename = 'voxel_distillation_comparison.json'
    else:
        modes = [('baseline18',18,False,False), ('shared18',18,True,False),
                 ('shared12',12,True,False), ('shared18_repeat',18,True,False),
                 ('baseline18_repeat',18,False,False)]
        filename = 'voxel_speed_comparison.json'
    for label,steps,shared,distilled in modes:
        torch.cuda.synchronize()
        t = perf_counter()
        depth = pipe.estimate_frame_depth(image) if shared else None
        outputs = [(name,reconstruct(rgba,hfer,steps,shared,depth,distilled)) for name,rgba,hfer in inputs]
        torch.cuda.synchronize()
        elapsed = perf_counter()-t
        matches = {}
        for name,out in outputs:
            # Compare native occupancy before adaptive downsampling as well as delivered cells.
            sets = []
            for field in ['coords_original','coords']:
                xyz = out[field][:,1:].cpu().numpy().astype(int)
                sets.append(set(map(tuple,xyz)))
            if label==modes[0][0]:
                reference[name]=sets
            matches[name]=[len(x & y)/len(x | y) for x,y in zip(sets,reference[name])]
        del outputs,depth
        report.append(dict(mode=label,seconds=elapsed,occupancy_iou=matches))
        (capture/filename).write_text(json.dumps(report,indent=2))
        print(label,elapsed,flush=True)
if __name__=='__main__':
    run()
