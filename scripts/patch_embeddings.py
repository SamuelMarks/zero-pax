import re

with open("src/zero_pax/praxis/layers/__init__.py") as f:
    content = f.read()

embeddings = {
    "Embedding": """
    def __call__(self, ids: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for Embedding.
        
        Args:
            ids (np.ndarray): Input ids.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        # simple mock
        return np.ones(ids.shape + (getattr(self, "input_dims", 1),))
""",
    "FullSoftmax": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for FullSoftmax.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        # simple mock
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
""",
    "GShardSharedEmbeddingSoftmax": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for GShardSharedEmbeddingSoftmax.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        # simple mock
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
""",
    "Ngrammer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for Ngrammer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "PositionalEmbedding": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for PositionalEmbedding.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "PositionalEmbedding2D": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for PositionalEmbedding2D.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "RandomVectorQuantizer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for RandomVectorQuantizer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "SharedEmbeddingSoftmax": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for SharedEmbeddingSoftmax.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        # simple mock
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
""",
    "TrainablePositionalEmbedding": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for TrainablePositionalEmbedding.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "VQNgrammer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for VQNgrammer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "VectorQuantization": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for VectorQuantization.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "VectorQuantizer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for VectorQuantizer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
}

for layer, math in embeddings.items():
    match = re.search(
        f"class {layer}\\(BaseModel\\):[\\s\\S]*?model_config = ConfigDict\\([^\\)]+\\)\\n(?:    .*\\n)*",
        content,
    )
    if match:
        block = match.group(0)
        # Re-attach __call__ after clearing it if present.
        block_clean = re.sub(r"    def __call__\(self.*?\n(?:        .*\n)*", "", block)
        content = content.replace(block, block_clean + "\n" + math.lstrip("\n"))

with open("src/zero_pax/praxis/layers/__init__.py", "w") as f:
    f.write(content)
