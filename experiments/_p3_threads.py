import time, torch, sys
from fes_bench import archs as A
net = A.build_rrdbnet_x4(); A.load_weights(net, "models/M23_realesrgan/RealESRGAN_x4plus.pth", "params_ema")
x = torch.rand(1,3,56,56)
for n in [1,2,4,8,12,16]:
    torch.set_num_threads(n)
    with torch.inference_mode():
        net(x); ts=[]
        for _ in range(3):
            t=time.perf_counter(); net(x); ts.append(time.perf_counter()-t)
    print(n, [round(v,2) for v in ts], flush=True)
