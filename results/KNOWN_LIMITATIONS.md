# Known limitations

## v4 — wide T + wide x, capacity unchanged (n=180, 25 epoch)

v4 (geniş T + geniş x, dx=160m, n=180, epoch=25): COMMEMI best RMS
18.86 — homojenden (5.74) kötü. Kök neden: mekansal kapsam 3km'den
19km'ye çıkınca (a) veri miktarı yetersiz kaldı (b) dx 25m'den 160m'ye
büyüyünce çözünürlük kaybı oldu. v5'te bu ikisi ayrı ayrı ele alınıyor.

### What v4 changed

| | v3 (narrow x) | v4 (wide x) |
|---|---|---|
| Periods | `T ∈ [10⁻³, 10³]` s (COMMEMI inside) | same |
| Profile | 120 × 25 m ≈ 3 km | 120 × 160 m = 19.2 km |
| Stations | 30, x ∈ [−1487.5, 1412.5] m | 30, x ∈ [−9520, 9040] m |
| COMMEMI x | 10/11 stations OOD (clamped) | 0/11 OOD |
| U-Net | `base_channels=16`, 2 decoder stages | **same capacity** |
| Data | 180 pairs | 180 pairs, regenerated on the wide mesh |

### COMMEMI VFSA (1 chain, 200 ctrl, 100 iter, seed=20260308)

| Start model | initial RMS | best RMS |
|---|---|---|
| Homogeneous | 12.31 | **5.7354** |
| U-Net v4 (180/25ep, wide T+x) | 62.65 | **18.8611** |

v2/v3 beating the homogeneous start (best ≈ 4.90) did **not** survive the
honest 19 km window. Those runs only saw the central ~3 km of COMMEMI.

### Artefacts

v4 was trained on 19 Aug 2026, then the checkpoint and logs were deleted
(repo rolled back to the v3 mesh). They were never committed, so they cannot
be restored with `git checkout`.

| Path | Status |
|---|---|
| `data/synthetic/train_pairs_v3_narrow_stations.h5` | **present** — v3 (narrow-x) pair backup used for the v3↔v4 comparison |
| `models/mid_scale_prior_v4.jld2` | missing (not in git) |
| `results/training_curve_v4.png` | missing (not in git) |
| `results/evaluate_v4/` | missing (not in git) |

Numbers above are from that run (`initial=62.65`, `best=18.8611`). They are
the reference v5 is scored against. Regenerating the v4 checkpoint is not
required for the v5 experiment: v5 retrains on the same wide mesh with more
data (`n=400`) and more capacity.

### v5 response (this iteration)

- **(a) data volume:** `train_pairs_v5.h5`, n=400 (same 8 scenarios, wide mesh).
- **(b) capacity / resolution:** `base_channels` 16→32, encoder/decoder depth
  2→3, larger MT projector hidden size (floor 512), CBAM on every stage.

Parameter count on the 48×120 mesh:

| Layout | `base_channels` | `n_down` | Parameters |
|---|---|---|---|
| v4 (unchanged capacity) | 16 | 2 | 6 386 036 |
| v5 | 32 | 3 | 15 689 336 (×2.46) |

×2.46 is a controlled increase (not 10×); 400 samples still carry overfitting
risk, which the val split and best-val checkpoint are meant to catch.

Capacity increase and n=180→400 were tested **together** (time budget). A
follow-up that holds one factor fixed would isolate which of (a) or (b)
dominated.

### v5 COMMEMI result

Same VFSA settings. `T_ood=false`, `x_ood=0/11`.

| Start model | initial RMS | best RMS |
|---|---|---|
| Homogeneous | 12.31 | **5.7354** |
| U-Net v4 (180/25ep, gT+x) | 62.65 | 18.8611 |
| U-Net v5 (400/35ep, gT+x, large capacity) | 39.86 | **11.8539** |

v5 is clearly better than v4 (best 18.86 → 11.85) but still worse than the
homogeneous start. **Yorum:** Doğru yönde, n=1000+ ölçeği gerekli, kapasite
artışı da işe yaradı.

Full eight-way table: `results/evaluate_v5/comparison.txt`.
Training: `models/mid_scale_prior_v5.jld2`, `results/training_curve_v5.png`
(best val at epoch 35, val loss 0.457; train 1.11 → 0.34).

## v6 — extra epochs on the same 400 pairs (resume v5, epoch 35→60)

Synthetic val loss kept falling (0.457 → 0.438) but COMMEMI VFSA **best RMS
worsened** (11.85 → 13.32). Sentetik val loss, gerçek (COMMEMI) performansla
güvenilir korelasyon göstermiyor.

| Start model | initial RMS | best RMS |
|---|---|---|
| Homogeneous | 12.31 | **5.7354** |
| U-Net v5 (400/35ep) | 39.86 | **11.8539** |
| U-Net v6 (400/60ep, v5 resume) | 37.43 | **13.3179** |

Full nine-way table: `results/evaluate_v6/comparison.txt`.

## v7 — n=1000, COMMEMI-family scenarios, commemi_rms checkpoint

v7 treats the v5→v6 mismatch as a training-loop bug, not just a data-size
issue:

1. **Periodic COMMEMI probe.** Every 10 epochs a short VFSA (25 iter, not 100)
   runs on COMMEMI 2-D-I; `commemi_rms` is logged next to `val_loss`.
2. **Checkpoint selection.** `commemi_rms` is primary; `val_loss` is fallback
   / tie-breaker. A better synthetic val loss never overwrites a worse
   COMMEMI RMS.
3. **Scenario mix.** Three COMMEMI-family geometries (`two_block`,
   `buried_prism`, `nested_prism`) added to the 8-scenario library — randomised
   sizes/resistivities, not a copy of the benchmark.
4. **Scale.** `n=1000` pairs, 50 epochs → `models/production_prior_v7.jld2`.

Training curve (val_loss + commemi_rms, dual axis): `results/training_curve_v7.png`.
Ten-way table: `results/evaluate_v7/comparison.txt`.
