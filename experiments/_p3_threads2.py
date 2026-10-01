import time, torch, sys
from fes_bench import adapters as AD
import numpy as np
n=int(sys.argv[1]); torch.set_num_threads(n)
cases=[("M03",(1,3,512,512),lambda net,x: net(x)[0]),("M05",(1,3,512,512),lambda net,x: net(x)["main_dec"]),
 ("M23",(1,3,224,224),lambda net,x: net(x)),("M24",(1,3,56,56),lambda net,x: net(x)),("M24",(1,3,224,224),lambda net,x: net(x)),
 ("M30",(1,3,224,224),lambda net,x: net(x)[0]),("M28-SIDD",(1,3,224,224),lambda net,x: net(x))]
for k,shp,f in cases:
    net,_=AD.load_net(k, verify_sha=False); x=torch.rand(shp)
    with torch.inference_mode():
        f(net,x); t=time.perf_counter(); f(net,x); dt=time.perf_counter()-t
    print(n,k,shp[-1],round(dt,2),flush=True)
