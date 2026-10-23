# Results

Generated 2026-10-23T09:03:42+00:00 by `make reproduce` from commit `2f50b92bc8a797a4503afb76885ac6fbfa1ce627`, Python 3.12.10 on Windows-10-10.0.19045-SP0.

All numbers come from this one run and are printed unrounded. Training is deterministic on one platform but not across platforms (see README, Phase 6), so a run on another OS gives slightly different values.

## How to reproduce

Requires Python 3.12, uv, Docker, and GNU make.

```bash
make reproduce
```

which runs, in order:

```bash
uv sync --locked
uv run overseer-train                  # retrain v1 from CryptoGuard's code
uv run overseer-drift                  # drift replay, steps 30-49
uv run overseer-retrain                # champion/challenger replay
docker build -t overseerml .           # image serves models/production.json
uv run python -m overseerml.loadtest   # load test against the container
uv run python -m overseerml.results    # writes this file
```

## 1. Champion (v1) PR-AUC before and after the drift

Pre-drift = steps 35-42, post-drift = steps 43-49 (dark-market shutdown at step 43). Labeled transactions of all steps in a window are pooled.

| Window | v1 PR-AUC |
|---|---|
| Pre-drift (35-42) | 0.9225811696943245 |
| Post-drift (43-49) | 0.053935333604891555 |
| Drop | 0.8686458360894329 |

## 2. When drift was first detected

**Input-drift detector (PSI): first flag at step 37**, 6 steps *before* the shutdown, while v1's hindsight PR-AUC that step was 0.892535249156892. Elliptic has no dates, only two-week time steps.

A step is flagged when more than 51 features have PSI > 0.25 against training steps 1-29, or its score PSI exceeds 0.4833592487041479 (both the highest seen in validation steps 30-34). Step 37 had 59 drifted features (score PSI 0.0449642338839286). All steps flagged: [37, 46]. The detector did **not** flag the shutdown step 43.

**Performance detector: first alert at step 46**, 3 steps after the shutdown, when lagged PR-AUC fell to 0.7517859128005614 (floor 0.8; labels arrive 2 steps late). This alert is what triggered retraining.

<details><summary>The 59 features with PSI &gt; 0.25 at step 37</summary>

| Feature | PSI |
|---|---|
| agg_43 | 8.283048714521238 |
| agg_46 | 8.283006562232506 |
| agg_44 | 8.263505288045648 |
| agg_8 | 7.921900076761664 |
| agg_10 | 7.745605990634468 |
| agg_7 | 7.720288038183531 |
| local_2 | 4.696900802344523 |
| agg_16 | 4.040913860925393 |
| agg_13 | 3.969451160780896 |
| agg_14 | 3.392452573410916 |
| agg_49 | 2.854509125165551 |
| agg_52 | 2.529055149254995 |
| agg_50 | 2.29083033945696 |
| agg_22 | 2.010604356564675 |
| agg_19 | 1.876930135218702 |
| local_3 | 1.7805979592341574 |
| agg_55 | 0.9906571148032858 |
| agg_58 | 0.9712796529442672 |
| agg_56 | 0.9235391636873502 |
| agg_20 | 0.8497634564113739 |
| agg_51 | 0.8115086442617565 |
| local_79 | 0.5080869379534204 |
| local_78 | 0.4556222345445579 |
| local_8 | 0.4454299270453131 |
| local_77 | 0.4432028855241582 |
| local_11 | 0.4056229292790252 |
| local_19 | 0.3993184916490518 |
| local_9 | 0.3918558262760643 |
| local_85 | 0.3910710135163899 |
| local_91 | 0.3786254814554267 |
| local_1 | 0.3784258046821764 |
| local_18 | 0.3773858336307535 |
| local_17 | 0.3727133416889487 |
| local_83 | 0.3684376801671184 |
| local_61 | 0.361701432477932 |
| local_67 | 0.3601973801350265 |
| local_84 | 0.346814707320597 |
| local_25 | 0.3443537271060402 |
| local_31 | 0.3437353503118695 |
| local_53 | 0.3395647585644559 |
| local_65 | 0.3351133997385751 |
| local_59 | 0.3350832985887239 |
| agg_1 | 0.3310926579093451 |
| agg_57 | 0.3301601321762321 |
| local_55 | 0.3204145806899916 |
| agg_38 | 0.3152755342934499 |
| local_29 | 0.3085841457463493 |
| local_23 | 0.3078319865779169 |
| local_28 | 0.3046673564368294 |
| local_22 | 0.3041259153488436 |
| agg_2 | 0.2993989130507908 |
| agg_4 | 0.2990727174935753 |
| local_89 | 0.291059337686793 |
| local_76 | 0.2812719223284793 |
| agg_39 | 0.2807688665439308 |
| agg_40 | 0.2795937164720345 |
| local_90 | 0.2722957563579994 |
| local_82 | 0.2637434709627883 |
| local_88 | 0.2505222852086287 |

</details>

## 3. Retrained challengers on the post-drift window

Each challenger is scored on the post-drift steps after its threshold window, and v1 on the same steps. Recovered share = (challenger - v1) / (v1 pre-drift - v1 on those steps). This is hindsight scoring: in the replay a challenger only serves steps after it is promoted.

| Challenger | Trained on | Scored on | Challenger PR-AUC | v1 PR-AUC | Recovered share of the drop | Promoted |
|---|---|---|---|---|---|---|
| retrain-s46 | steps 1-40 | 43-49 | 0.2698573070967563 | 0.053935333604891555 | 0.2485730829769777 | yes |
| retrain-s47 | steps 1-41 | 44-49 | 0.19219105323966415 | 0.05656127957697825 | 0.15661276976480065 | no |
| retrain-s48 | steps 1-42 | 45-49 | 0.33936451819805447 | 0.07947104101785991 | 0.30825566950332084 | no |
| retrain-s49 | steps 1-43 | 46-49 | 0.49711563344518206 | 0.1286830226276528 | 0.4640804518549761 | yes |

## 4. Promotion decisions

A challenger is promoted only if its PR-AUC on the comparison steps (the two newest labeled steps, which neither model trained on) beats the serving model's by at least 0.02.

| Trigger step | Serving model | Challenger | Compared on | Illicit in comparison | Serving PR-AUC | Challenger PR-AUC | Challenger - serving | Promoted |
|---|---|---|---|---|---|---|---|---|
| 46 | v1 | retrain-s46 | steps 43-44 | 48 | 0.04173935578018027 | 0.175995153880249 | 0.1342557981000687 | yes |
| 47 | retrain-s46 | retrain-s47 | steps 44-45 | 29 | 0.10615170680296954 | 0.06606094527628535 | -0.040090761526684185 | no |
| 48 | retrain-s46 | retrain-s48 | steps 45-46 | 7 | 0.09427196680660561 | 0.07804198926083361 | -0.016229977545772004 | no |
| 49 | retrain-s46 | retrain-s49 | steps 46-47 | 24 | 0.24261448060746374 | 0.3229629184628423 | 0.08034843785537857 | yes |

Production model after the replay: `retrain-s49`.

## 5. API latency

`POST /predict` against the Docker image serving `retrain-s49`: 50 sequential warm-up requests (excluded), then 2000 requests from 8 concurrent client threads, each a real transaction sampled from test steps 35-49. Client and server ran on the same machine.

| Measure | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | max (ms) |
|---|---|---|---|---|---|
| Client round trip | 72.7352000103565 | 438.09391499671545 | 1162.0314840052743 | 131.85495140019339 | 3411.214199993992 |
| Server (scoring + SHAP) | 4.1048064999813505 | 5.1280096999732905 | 6.123269000015625 | 4.217490097000053 | 8.166938000044865 |

Throughput: 60.38469598309339 requests/s (2000 requests in 33.120974899997236 s). Server-logged requests after warm-up: 2000.

Hardware: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz, 20 logical CPUs, 68616658944 bytes RAM, Windows-10-10.0.19045-SP0. Docker 29.8.2 (Docker Desktop) with 20 CPUs and 33602535424 bytes of memory available to containers.
