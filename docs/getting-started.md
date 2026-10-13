# Getting started

```bash
pip install tavan            # core, NumPy only
pip install "tavan[plot]"    # + matplotlib
```

## The `TimeSeries` container

Everything starts from a `TimeSeries` — a regularly sampled, possibly
multivariate series with timestamps.

```python
import tavan
from tavan import TimeSeries

# from an array
ts = TimeSeries(values, start="2024-01-01", freq="1h", names=["load"], unit="MW")

# from a CSV (no pandas required)
ts = TimeSeries.from_csv("load.csv", time_col="timestamp", value_cols=["load"])

# a demo series
ts = tavan.make_load(n_days=365, with_temperature=True)
```

Preprocessing methods return new series and remember their transforms:

```python
clean = ts.clean_outliers().impute("seasonal").scale("robust")
back  = clean.inverse_transform(clean.values)      # undo scaling
X, Y  = ts.windows(lookback=96, horizon=24)         # supervised windows
profiles = ts.to_profiles(period=24)                # (n_days, 24, C)
```

## Persisting models

Every fitted model saves to a single, pickle-free `.npz` archive:

```python
gen.save("model.npz")
gen = tavan.load("model.npz")
```
