# BBOBax

**BBOBax: Accelerated Black-Box Optimization Benchmark in JAX.**

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for installation:

```bash
# Clone the repository
git clone https://github.com/maxencefaldor/bbobax.git
cd bbobax

# Create a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies and the package
uv pip install -e .
```

## Usage

Here is a basic example of how to use BBOBax:

```python
import jax
from bbobax import BBOB

# Initialize BBOB task with default functions
bbob = BBOB.create_default(min_num_dims=2, max_num_dims=10)

# Sample a task instance
key = jax.random.key(0)
key_task, key_init, key_eval, key_x = jax.random.split(key, 4)
task_params = bbob.sample(key_task)

# Initialize state
state = bbob.init(key_init, task_params)

# Sample solution
x = bbob.sample_x(key_x)

# Evaluate
state, eval_metrics = bbob.evaluate(key_eval, x, state, task_params)

print(f"Fitness: {eval_metrics.fitness}")
```
