from abc import ABC, abstractmethod
import jax.numpy as jnp

class BaseModel(ABC):
    """
    Generic interface for all scientific models (ODE, Stochastic, etc.).
    This allows the SBI Engine to run simulations and the LLM Agent to reason about the model.
    """
    
    @abstractmethod
    def simulate(self, params, time_points, y0):
        pass

    @abstractmethod
    def get_parameter_metadata(self):
        pass

    @abstractmethod
    def get_latex(self):
        pass

    @abstractmethod
    def get_initial_conditions(self):
        pass
