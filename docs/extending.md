# Extending tavan

Every model is registered under a name and discoverable via `tavan.available()`.

```python
from tavan.generate import BaseGenerator
from tavan.registry import register

@register("generator", "my_model")
class MyModel(BaseGenerator):
    def __init__(self, strength=1.0, seed=0):
        self.strength = strength
        self.seed = seed
    def _fit(self, X, cond):
        self.mu_ = X.mean(0)
    def _sample(self, n, cond, rng):
        return self.mu_ + rng.normal(0, self.strength, (n, self.n_features_))

# usable everywhere by name
import tavan
tavan.ScenarioGenerator("my_model", model_params={"strength": 2.0}).fit(data)
```

Declaring `_state_attrs` makes the model saveable with `.save()` / `tavan.load()`.

## Plugins

External packages can contribute models through the `tavan.plugins` entry-point
group, loaded lazily the first time the registry is queried:

```toml
[project.entry-points."tavan.plugins"]
my_pkg = "my_pkg.register_models"
```
