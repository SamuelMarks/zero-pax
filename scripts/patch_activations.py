import re

with open("src/zero_pax/praxis/layers/__init__.py") as f:
    content = f.read()

# Define the activations to replace
activations = {
    "CubedReLU": "np.power(np.maximum(x, 0.0), 3)",
    "ELU": "np.where(x > 0, x, 1.0 * (np.exp(x) - 1))",
    "GELU": "0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))",
    "LeakyReLU": "np.where(x > 0, x, self.negative_slope * x)",
    "ReLU": "np.maximum(x, 0.0)",
    "ReLU6": "np.minimum(np.maximum(x, 0.0), 6.0)",
    "SiLU": "x * (1 / (1 + np.exp(-x)))",
    "Sigmoid": "1 / (1 + np.exp(-x))",
    "SquaredReLU": "np.square(np.maximum(x, 0.0))",
    "Swish": "x * (1 / (1 + np.exp(-x)))",
    "Tanh": "np.tanh(x)",
}

for act, math in activations.items():
    # find the class block
    match = re.search(
        f"class {act}\\(BaseModel\\):[\\s\\S]*?model_config = ConfigDict\\([^\\)]+\\)\\n(?:    .*\\n)*",
        content,
    )
    if match:
        block = match.group(0)
        # add __call__
        new_call = f'''
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for {act}.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        """
        return {math}
'''
        content = content.replace(block, block + new_call)
    else:
        print(f"Could not find class {act}")

with open("src/zero_pax/praxis/layers/__init__.py", "w") as f:
    f.write(content)
