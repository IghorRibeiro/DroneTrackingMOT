# Cinemática por Evento de Tracking

Pipeline: HardMining + SAHI512 (BoT‑SORT)
Total de transições analisadas: 464345

## TP

| Métrica | count | mean | std | median | p90 | p95 | p99 | p99.5 | p99.9 |
|---------|-------|------|-----|--------|-----|-----|-----|-------|-------|
| area_ratio | 421929 | 1.0007 | 0.0388 | 1.0000 | 1.0337 | 1.0528 | 1.1212 | 1.1615 | 1.2976 |
| width_ratio | 421929 | 1.0003 | 0.0225 | 1.0000 | 1.0181 | 1.0282 | 1.0703 | 1.0934 | 1.1763 |
| height_ratio | 421929 | 1.0002 | 0.0238 | 1.0000 | 1.0203 | 1.0312 | 1.0742 | 1.0995 | 1.1741 |
| center_shift_normalized | 421929 | 0.0016 | 0.0022 | 0.0010 | 0.0035 | 0.0048 | 0.0112 | 0.0176 | 0.0204 |
| aspect_ratio_ratio | 421929 | 1.0005 | 0.0257 | 1.0000 | 1.0222 | 1.0343 | 1.0775 | 1.1027 | 1.2004 |

## FP

| Métrica | count | mean | std | median | p90 | p95 | p99 | p99.5 | p99.9 |
|---------|-------|------|-----|--------|-----|-----|-----|-------|-------|
| area_ratio | 42150 | 1.0014 | 0.0571 | 0.9997 | 1.0552 | 1.0836 | 1.1677 | 1.2252 | 1.4062 |
| width_ratio | 42150 | 1.0004 | 0.0325 | 1.0000 | 1.0292 | 1.0482 | 1.0972 | 1.1240 | 1.2179 |
| height_ratio | 42150 | 1.0006 | 0.0351 | 1.0000 | 1.0317 | 1.0503 | 1.1053 | 1.1374 | 1.2306 |
| center_shift_normalized | 42150 | 0.0022 | 0.0019 | 0.0019 | 0.0045 | 0.0057 | 0.0075 | 0.0086 | 0.0175 |
| aspect_ratio_ratio | 42150 | 1.0006 | 0.0372 | 0.9996 | 1.0332 | 1.0522 | 1.1105 | 1.1540 | 1.3062 |

## ID_SWITCH

| Métrica | count | mean | std | median | p90 | p95 | p99 | p99.5 | p99.9 |
|---------|-------|------|-----|--------|-----|-----|-----|-------|-------|
| area_ratio | 266 | 1.0058 | 0.1237 | 1.0008 | 1.1337 | 1.1944 | 1.3547 | 1.4195 | 1.4496 |
| width_ratio | 266 | 0.9966 | 0.0699 | 0.9993 | 1.0652 | 1.0987 | 1.2077 | 1.2492 | 1.2996 |
| height_ratio | 266 | 1.0067 | 0.0724 | 1.0018 | 1.0882 | 1.1376 | 1.2173 | 1.2565 | 1.3210 |
| center_shift_normalized | 266 | 0.0016 | 0.0020 | 0.0011 | 0.0038 | 0.0044 | 0.0087 | 0.0109 | 0.0172 |
| aspect_ratio_ratio | 266 | 0.9924 | 0.0688 | 0.9932 | 1.0633 | 1.0978 | 1.2056 | 1.2286 | 1.2668 |

## FN (não gera transições, count=0)

## Comparação Estatística (Mann‑Whitney U)

### TP vs FP

| Métrica | p-valor | Mediana TP | Mediana FP | Média TP | Média FP |
|---------|---------|------------|--------------|----------|------------|
| area_ratio | 9.27e-01 | 1.0000 | 0.9997 | 1.0007 | 1.0014 |
| width_ratio | 1.08e-02 | 1.0000 | 1.0000 | 1.0003 | 1.0004 |
| height_ratio | 4.76e-01 | 1.0000 | 1.0000 | 1.0002 | 1.0006 |
| center_shift_normalized | 0.00e+00 | 0.0010 | 0.0019 | 0.0016 | 0.0022 |
| aspect_ratio_ratio | 3.44e-04 | 1.0000 | 0.9996 | 1.0005 | 1.0006 |

### TP vs ID_SWITCH

| Métrica | p-valor | Mediana TP | Mediana ID_SWITCH | Média TP | Média ID_SWITCH |
|---------|---------|------------|--------------|----------|------------|
| area_ratio | 3.99e-01 | 1.0000 | 1.0008 | 1.0007 | 1.0058 |
| width_ratio | 9.09e-01 | 1.0000 | 0.9993 | 1.0003 | 0.9966 |
| height_ratio | 2.69e-01 | 1.0000 | 1.0018 | 1.0002 | 1.0067 |
| center_shift_normalized | 2.07e-01 | 0.0010 | 0.0011 | 0.0016 | 0.0016 |
| aspect_ratio_ratio | 2.08e-03 | 1.0000 | 0.9932 | 1.0005 | 0.9924 |

## Conclusão Preliminar

**Variáveis com diferença altamente significativa (p<0.001):**
- center_shift_normalized (TP vs FP)
- aspect_ratio_ratio (TP vs FP)

Essas variáveis são promissoras para incluir no custo de associação do BoT‑SORT.
