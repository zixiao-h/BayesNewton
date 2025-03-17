import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "METAL")
print(jax.devices())

a = jnp.array([1.0, 2.0, 3.0])
print(a)
