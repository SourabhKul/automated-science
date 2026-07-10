import jax.numpy as jnp

logic_code_string = """
def dynamics(t, y, args):
    return jnp.clip(jnp.array([1.0, 2.0]), -1e6, 1e6)
"""

full_module = f"""import jax.numpy as jnp
from diffrax import SaveAt

{logic_code_string}

class CandidateModel:
    def get_parameter_metadata(self):
        return []
    def simulate(self):
        return dynamics(0, jnp.array([0,0]), jnp.array([0]))
"""

import importlib.util
spec = importlib.util.spec_from_loader("test_mod", loader=None)
mod = importlib.util.module_from_spec(spec)
try:
    exec(full_module, mod.__dict__)
    model = mod.CandidateModel()
    print("Simulate:", model.simulate())
except Exception as e:
    print("ERROR:", type(e), e)
