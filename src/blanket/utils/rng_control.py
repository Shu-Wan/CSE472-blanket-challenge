"""
Project-wide random sampling utilities with unified RNG control.

This module provides a unified interface for random number generation across
different libraries (Python stdlib, NumPy, PyTorch) to ensure reproducibility
and consistent seeding throughout the project.
"""

import random
from typing import Optional

import numpy as np
import torch


class GlobalRNG:
    """
    Global random number generator controller for project-wide reproducibility.

    This class provides a unified interface to control random seeds across
    Python's random module, NumPy, and PyTorch (if available).
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the global RNG controller.

        Args:
            seed: Random seed to use. If None, uses system entropy.
        """
        self._seed = seed
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for all available RNG libraries.

        Args:
            seed: Random seed value
        """
        self._seed = seed

        # Set Python stdlib random seed for callers that rely on the global state
        random.seed(seed)

        # Set PyTorch seed if available
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic CUDA operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_seed(self) -> Optional[int]:
        """Get the current seed value."""
        return self._seed

    def numpy_rng(self, seed: Optional[int] = None) -> np.random.Generator:
        """
        Get a NumPy Generator instance.

        Args:
            seed: Optional seed for this specific RNG instance.
                  If None, uses a derived seed from the global seed.

        Returns:
            NumPy Generator instance (np.random.Generator)
        """
        if seed is None and self._seed is not None:
            # Derive a seed from the global seed
            seed = (self._seed + hash("numpy_rng")) % (2**32)

        return np.random.default_rng(seed)

    def torch_generator(self, seed: Optional[int] = None) -> torch.Generator:
        """
        Get a PyTorch Generator instance.

        Args:
            seed: Optional seed for this specific generator.
                  If None, uses a derived seed from the global seed.

        Returns:
            PyTorch Generator instance, or None if PyTorch not available
        """

        if seed is None and self._seed is not None:
            # Derive a seed from the global seed
            seed = (self._seed + hash("torch_generator")) % (2**32)

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        return generator

    def reset(self) -> None:
        """Reset all RNGs to use system entropy (non-deterministic)."""
        self._seed = None

        # Reset Python stdlib to entropy
        random.seed()

        # Do not call `np.random.seed()` to avoid mutating global NumPy state.
        # Consumers should create fresh `Generator` instances via `get_numpy_rng()`.

        torch.seed()


# Global instance for project-wide use
global_rng = GlobalRNG()


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for all RNG libraries.

    Args:
        seed: Random seed value

    Example:
        >>> from blanket.utils.rng_control import set_global_seed
        >>> set_global_seed(42)
        >>> # Now all random operations are deterministic
    """
    global_rng.set_seed(seed)


def get_global_seed() -> Optional[int]:
    """
    Get the current global random seed.

    Returns:
        Current seed value, or None if using entropy
    """
    return global_rng.get_seed()


def get_numpy_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a NumPy Generator instance with optional seed.

    Args:
        seed: Optional seed. If None, derives from global seed.

    Returns:
        NumPy Generator instance

    Example:
        >>> from blanket.utils.rng_control import get_numpy_rng
        >>> rng = get_numpy_rng(42)
        >>> random_array = rng.normal(0, 1, size=100)
    """
    return global_rng.numpy_rng(seed)


def get_torch_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Get a PyTorch Generator instance with optional seed.

    Args:
        seed: Optional seed. If None, derives from global seed.

    Returns:
        PyTorch Generator instance, or None if PyTorch not available

    Example:
        >>> from blanket.utils.rng_control import get_torch_generator
        >>> generator = get_torch_generator(42)
        >>> if generator is not None:
        ...     tensor = torch.randn(100, generator=generator)
    """
    return global_rng.torch_generator(seed)


def with_temp_seed(seed: int):
    """
    Context manager for temporary seed setting.

    Args:
        seed: Temporary seed to use

    Example:
        >>> from blanket.utils.rng_control import with_temp_seed, global_rng
        >>> global_rng.set_seed(42)
        >>> with with_temp_seed(123):
        ...     # Operations here use seed 123
        ...     result = np.random.random()
        >>> # Back to seed 42 here
    """

    class TempSeedContext:
        """Context manager for temporarily setting a random seed."""

        def __init__(self, new_seed: int):
            self.new_seed = new_seed
            self.old_seed = None

        def __enter__(self):
            self.old_seed = global_rng.get_seed()
            global_rng.set_seed(self.new_seed)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.old_seed is not None:
                global_rng.set_seed(self.old_seed)
            else:
                global_rng.reset()

    return TempSeedContext(seed)


# Example usage and testing functions
def _example_usage():
    """Example usage of the random utilities."""
    print("=== Blanket Random Utilities Example ===")

    # Set global seed for reproducibility
    print("\n1. Setting global seed to 42...")
    set_global_seed(42)

    # Use with different libraries
    print(f"Global seed: {get_global_seed()}")

    print("\n2. Python stdlib random:")
    print(f"random.random(): {random.random()}")
    print(f"random.randint(1, 10): {random.randint(1, 10)}")

    print("\n3. NumPy random:")
    print(f"np.random.random(): {np.random.random()}")
    print(f"np.random.normal(): {np.random.normal()}")

    # Get dedicated RNG instances
    print("\n4. Dedicated NumPy RNG:")
    rng = get_numpy_rng(seed=123)
    print(f"rng.normal(0, 1, 3): {rng.normal(0, 1, 3)}")

    # PyTorch if available
    print("\n5. PyTorch random:")
    print(f"torch.randn(3): {torch.randn(3)}")

    generator = get_torch_generator(seed=456)
    print(f"torch.randn(3, generator=gen): {torch.randn(3, generator=generator)}")

    # Temporary seed context
    print("\n6. Temporary seed context:")
    print(f"Before context: {np.random.random()}")

    with with_temp_seed(999):
        print(f"Inside context (seed=999): {np.random.random()}")
        print(f"Inside context: {np.random.random()}")

    print(f"After context: {np.random.random()}")


if __name__ == "__main__":
    _example_usage()
