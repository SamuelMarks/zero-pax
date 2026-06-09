import re

with open("src/zero_pax/praxis/layers/__init__.py") as f:
    content = f.read()

act = "SigmoidCrossEntropy"
match = re.search(
    f"class {act}\\(BaseModel\\):[\\s\\S]*?model_config = ConfigDict\\([^\\)]+\\)\\n(?:    .*\\n)*",
    content,
)
if match:
    block = match.group(0)
    new_call = f'''
    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Forward pass for {act}.
        
        Args:
            logits (np.ndarray): Input logits.
            labels (np.ndarray): Target labels.
            
        Returns:
            np.ndarray: Loss array.
        """
        # numerically stable sigmoid cross entropy
        # z = logits, x = labels
        # max(x, 0) - x * z + log(1 + exp(-abs(x)))
        return np.maximum(logits, 0) - logits * labels + np.log(1 + np.exp(-np.abs(logits)))
'''
    content = content.replace(block, block + new_call)
else:
    print(f"Could not find class {act}")

with open("src/zero_pax/praxis/layers/__init__.py", "w") as f:
    f.write(content)
