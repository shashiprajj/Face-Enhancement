#!/bin/sh
# R5-10: the unchanged R5-6 harness (scripts/r5_6_service_eval.py) against 3 service replicas started with
# FES_NAFNET_PROFILE=r5_10 (the FROZEN R5-10 candidate; see scripts/r5_10_services.sh). Run ONCE, inside the client
# container on network fes-r510:
#   MSYS_NO_PATHCONV=1 docker run -d --name r510-client --network fes-r510 -e PYTHONPATH=src -e FES_GIT_HEAD=<sha> \
#     -v "<repo>:/workspace" -w /workspace fes-dev:cpu-p5 sh scripts/r5_10_validate.sh
# Evaluation only — no threshold or config may change after these results are seen.
retry() { for a in 1 2 3 4 5 6 7 8 9 10; do "$@" && return 0; echo "RETRY $a: $*"; sleep 20; done; return 1; }
S=http://fes-r510-s1:8080,http://fes-r510-s2:8080,http://fes-r510-s3:8080
# every replica must be READY (body "ready": true, not just HTTP 200) with the frozen R5-10 seal — otherwise abort:
# a replica whose model load failed answers 503 NOT_READY instantly (first attempt: s3 hit ENOMEM loading NAFNet)
SEAL=$(python -c "import json;print(json.load(open('configs/phase5/r5_10_nafnet_candidate.json'))['sha256'])")
ok=0
for i in $(seq 1 120); do python -c "import httpx,sys;d=[httpx.get(f'http://fes-r510-s{k}:8080/ready').json() for k in (1,2,3)];sys.exit(0 if all(x['ready'] and x['nafnet_profile']=='r5_10' and x['seals']['nafnet_route']=='$SEAL' for x in d) else 1)" 2>/dev/null && ok=1 && break; sleep 5; done
[ "$ok" = 1 ] || { echo "ABORT: not all replicas ready with the R5-10 seal"; exit 2; }
python -c "import httpx,json;[print(json.dumps({k:httpx.get(f'http://fes-r510-s{s}:8080/ready').json()[k] for k in ('ready','route_mode','nafnet_profile','seals','nafnet_model')})) for s in (1,2,3)]"
retry python scripts/r5_6_service_eval.py run --dataset ztest --mode face_chip --name r510_ztest_face_chip --services $S
retry python scripts/r5_6_service_eval.py run --dataset ztest --mode full_image --name r510_ztest_full_image --services $S
retry python scripts/r5_6_service_eval.py run --dataset real --mode full_image --name r510_real_full_image --services $S --inflight 1
retry python scripts/r5_6_service_eval.py run --dataset test --mode face_chip --name r510_test_face_chip --services $S
retry python scripts/r5_6_service_eval.py run --dataset test --mode full_image --name r510_test_full_image --services $S
python -c "import httpx;[print('restarts', s, httpx.get(f'http://fes-r510-s{s}:8080/ready').json()['nafnet_worker_restarts'],httpx.get(f'http://fes-r510-s{s}:8080/ready').json()['worker_restarts']) for s in (1,2,3)]"
echo SERVICE_RUNS_DONE
for n in r510_test_face_chip r510_test_full_image r510_ztest_face_chip r510_ztest_full_image; do
  python scripts/r5_6_service_eval.py identity --name $n
done
echo R510_VALIDATION_DONE
