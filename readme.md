# mtlab — Material Testing Lab (arterial tissue)

Clean, modular toolkit for:
- Loading material test data
- Fitting constitutive models
- Summarizing parameters by group/region
- Computing strain energy metrics
- Verification/validation of constitutive models
- Publication-ready plots

## Install (editable)
```bash
pip install -e .
```

## Quickstart
```bash
mt fit --data data/material.csv --model ogden2 --out out/
mt summarize --fits out/fits.csv --out out/
mt verify --data data/material.csv --models ogden2,mooney --kfold 5 --out out/verify
mt plot-se --stats out/strain_energy_stats.csv --out out/figs
```
