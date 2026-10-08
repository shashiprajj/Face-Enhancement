#!/bin/sh
# R5-10: continuation of scripts/r5_10_validate.sh on 4 replicas (fes-r510-s4 added mid-run to use idle CPU; same
# frozen image/config — the readiness check below verifies the R5-10 seal on all four). The harness resumes from
# records.jsonl, so completed requests are not re-sent. Evaluation only.
retry() { for a in 1 2 3 4 5 6 7 8 9 10; do "$@" && return 0; echo "RETRY $a: $*"; sleep 20; done; return 1; }
S=http://fes-r510-s1:8080,http://fes-r510-s2:8080,http://fes-r510-s3:8080,http://fes-r510-s4:8080
SEAL=$(python -c "import json;print(json.load(open('configs/phase5/r5_10_nafnet_candidate.json'))['sha256'])")
python -c "import httpx,sys;d=[httpx.get(f'http://fes-r510-s{k}:8080/ready').json() for k in (1,2,3,4)];sys.exit(0 if all(x['ready'] and x['nafnet_profile']=='r5_10' and x['seals']['nafnet_route']=='$SEAL' for x in d) else 1)" || { echo "ABORT: not all replicas ready with the R5-10 seal"; exit 2; }
retry python scripts/r5_6_service_eval.py run --dataset test --mode face_chip --name r510_test_face_chip --services $S
retry python scripts/r5_6_service_eval.py run --dataset test --mode full_image --name r510_test_full_image --services $S
python -c "import httpx;[print('restarts', s, httpx.get(f'http://fes-r510-s{s}:8080/ready').json()['nafnet_worker_restarts'],httpx.get(f'http://fes-r510-s{s}:8080/ready').json()['worker_restarts']) for s in (1,2,3,4)]"
echo SERVICE_RUNS_DONE
for n in r510_test_face_chip r510_test_full_image r510_ztest_face_chip r510_ztest_full_image; do
  python scripts/r5_6_service_eval.py identity --name $n
done
echo R510_VALIDATION_DONE
