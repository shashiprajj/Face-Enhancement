import sys, time, torch, cv2
from multiprocessing import get_context
def work(args):
    th, k = args
    torch.set_num_threads(th); cv2.setNumThreads(1)
    from fes_bench.adapters import load_net
    net,_ = load_net("M23", verify_sha=False); x = torch.rand(1,3,112,112)
    with torch.inference_mode():
        net(x); t=time.perf_counter()
        for _ in range(k): net(x)
    return time.perf_counter()-t
if __name__ == "__main__":
    for spec in sys.argv[1:]:
        P, th = map(int, spec.split("x")); k=4
        with get_context("spawn").Pool(P) as pool:
            t0=time.perf_counter(); ts = pool.map(work, [(th,k)]*P); wall=time.perf_counter()-t0
        print(f"{P}x{th}: per-img {max(ts)/k:.2f}s  throughput {P*k/max(ts):.3f} img/s", flush=True)
