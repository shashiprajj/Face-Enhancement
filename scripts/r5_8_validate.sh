#!/bin/sh
# R5-8: the unchanged R5-6 harness against the R5-7 service build (3 replicas, default config)
retry() { for a in 1 2 3 4 5 6 7 8 9 10; do "$@" && return 0; echo "RETRY $a: $*"; sleep 20; done; return 1; }
S=http://fes-r56-s1:8080,http://fes-r56-s2:8080,http://fes-r56-s3:8080
for i in $(seq 1 90); do python -c "import httpx,sys;sys.exit(0 if all(httpx.get(f'http://fes-r56-s{k}:8080/ready').status_code==200 for k in (1,2,3)) else 1)" 2>/dev/null && break; sleep 5; done
python -c "import httpx,json;[print(json.dumps({k:httpx.get(f'http://fes-r56-s{s}:8080/ready').json()[k] for k in ('ready','route_mode','nafnet_profile','seals','nafnet_model')})) for s in (1,2,3)]"
retry python scripts/r5_6_service_eval.py run --dataset ztest --mode face_chip --name r58_ztest_face_chip --services $S
retry python scripts/r5_6_service_eval.py run --dataset ztest --mode full_image --name r58_ztest_full_image --services $S
retry python scripts/r5_6_service_eval.py run --dataset real --mode full_image --name r58_real_full_image --services $S --inflight 1
retry python scripts/r5_6_service_eval.py run --dataset test --mode face_chip --name r58_test_face_chip --services $S
retry python scripts/r5_6_service_eval.py run --dataset test --mode full_image --name r58_test_full_image --services $S
python -c "import httpx;[print(httpx.get(f'http://fes-r56-s{s}:8080/ready').json()['nafnet_worker_restarts'],httpx.get(f'http://fes-r56-s{s}:8080/ready').json()['worker_restarts']) for s in (1,2,3)]"
echo RERUN_DONE
