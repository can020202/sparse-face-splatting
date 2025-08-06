
import os
import numpy as np
import torch
import torchvision
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R_scipy, Slerp

from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render

def make_cam(Rm, t, fovx, fovy, W, H):
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    def to_cuda(mat):
        return torch.as_tensor(mat, dtype=torch.float32, device="cuda")
    P  = to_cuda(getProjectionMatrix(0.01, 100.0, fovx, fovy)).T
    WV = to_cuda(getWorld2View2(Rm, t, np.zeros(3), 1.0)).T
    cam = type("Cam", (), {})()
    cam.world_view_transform = WV
    cam.projection_matrix    = P
    cam.full_proj_transform  = WV @ P
    cam.camera_center        = torch.inverse(WV)[:3,3]
    cam.image_width, cam.image_height = W, H
    cam.FoVx, cam.FoVy = fovx, fovy
    return cam

def main():

    # CLI-Args

    parser = ArgumentParser(description="Render smooth interpolation between original cameras")
    model_params    = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iters", type=int,   default=-1,
                        help="Trainingsiteration (z.B. 1550; -1 = neueste)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Skalierungsfaktor (1.0 = Originalauflösung)")
    parser.add_argument("--steps", type=int,   default=5,
                        help="Anzahl der Zwischenschritte zwischen JEDEM Kamerapaar")
    parser.add_argument("--only_train", action="store_true",
                        help="nur Trainings-Views (default)")
    parser.add_argument("--only_test",  action="store_true",
                        help="nur Test-Views")
    parser.add_argument("--no_shuffle", action="store_true",
                        help="kein Shuffle der Original-Views")
    args = get_combined_args(parser)
    if args.no_shuffle: 
        torch.manual_seed(0)


    # GaussianModel & Scene

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(
        args,
        gaussians,
        load_iteration=args.iters,
        shuffle=not args.no_shuffle,
        resolution_scales=[max(args.scale, 1.0)]
    )
    bgcol = [1.0,1.0,1.0] if args.white_background else [0.0,0.0,0.0]
    background = torch.tensor(bgcol, dtype=torch.float32, device="cuda")


    # Original Views
    res = max(args.scale, 1.0)
    if args.only_test and not args.only_train:
        orig_views = scene.getTestCameras(res)
    else:
        orig_views = scene.getTrainCameras(res)

    pipeline_obj = pipeline_params.extract(args)


    # View-List interpolated

    interp_views = []
    Nsteps = args.steps
    for i in range(len(orig_views)-1):
        v0 = orig_views[i]
        v1 = orig_views[i+1]
        # Rotation & Translation
        R0 = v0.R
        R1 = v1.R
        t0 = v0.T
        t1 = v1.T

        # SLERP-Interpolator
        key_times = [0.0, 1.0]
        rots      = R_scipy.from_matrix([R0, R1])
        slerp     = Slerp(key_times, rots)

        alphas = np.linspace(0.0, 1.0, Nsteps+2)
        for a in alphas[:-1]:
            Rm = slerp(a).as_matrix()
            tm = (1-a)*t0 + a*t1
            # new cam object
            cam = make_cam(
                Rm, tm,
                v0.FoVx, v0.FoVy,
                v0.image_width, v0.image_height
            )
            interp_views.append(cam)

    interp_views.append(
        make_cam(
            orig_views[-1].R,
            orig_views[-1].T,
            orig_views[-1].FoVx,
            orig_views[-1].FoVy,
            orig_views[-1].image_width,
            orig_views[-1].image_height
        )
    )


    # Render-Loop

    out_dir = os.path.join(args.model_path, f"renders_interpolated_{args.iters}")
    os.makedirs(out_dir, exist_ok=True)

    for idx, view in enumerate(interp_views):
        out = render(view, gaussians, pipeline_obj, background)["render"]
        torchvision.utils.save_image(
            out,
            os.path.join(out_dir, f"{idx:03d}.png")
        )

if __name__ == "__main__":
    main()
