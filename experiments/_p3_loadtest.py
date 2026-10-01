import time, torch
from fes_bench import archs as A
specs = [
 ("rfpp", A.build_restoreformerpp, "models/M03_restoreformerpp/RestoreFormer++.ckpt", "state_dict", "vqvae.", False, (1,3,512,512)),
 ("vqfr", A.build_vqfr_v1, "models/M05_vqfr/VQFR_v1-33a1fac5.pth", "params", None, True, (1,3,512,512)),
 ("resrgan", A.build_rrdbnet_x4, "models/M23_realesrgan/RealESRGAN_x4plus.pth", "params_ema", None, True, (1,3,56,56)),
 ("swinir", A.build_swinir_realsr_l_x4, "models/M24_swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth", "params_ema", None, True, (1,3,56,56)),
 ("fbcnn", A.build_fbcnn_color, "models/M30_fbcnn/fbcnn_color.pth", None, None, True, (1,3,224,224)),
 ("naf_sidd", A.build_nafnet_sidd, "models/M28_nafnet/NAFNet-SIDD-width64.pth", "params", None, True, (1,3,224,224)),
 ("naf_gopro", A.build_nafnet_gopro, "models/M28_nafnet/NAFNet-GoPro-width64.pth", "params", None, True, (1,3,224,224)),
]
torch.set_num_threads(16)
for name, b, p, key, pre, strict, shp in specs:
    net = b(); r = A.load_weights(net, p, key, pre, strict)
    x = torch.rand(shp)
    with torch.inference_mode():
        y = net(x); t=time.perf_counter(); y = net(x); dt=time.perf_counter()-t
    y0 = y[0] if isinstance(y,(tuple,list)) else (y['main_dec'] if isinstance(y,dict) else y)
    print(name, "missing", len(r["missing_keys"]), r["missing_keys"][:3], "unexpected", len(r["unexpected_keys"]), r["unexpected_keys"][:3], "params", r["n_model_params"], "out", tuple(y0.shape), f"{dt:.2f}s", flush=True)
