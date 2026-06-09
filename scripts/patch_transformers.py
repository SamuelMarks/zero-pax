import re

with open("src/zero_pax/praxis/layers/__init__.py") as f:
    content = f.read()

transformers = {
    "AdaptedTransformerFeedForward": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for AdaptedTransformerFeedForward.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "FeedForward": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for FeedForward.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "PipelinedTransformer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for PipelinedTransformer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "SSMTransformer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for SSMTransformer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "StackedTransformer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for StackedTransformer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "StackedTransformerRepeated": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for StackedTransformerRepeated.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "Transformer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for Transformer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "TransformerEncoderDecoder": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for TransformerEncoderDecoder.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "TransformerFeedForward": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for TransformerFeedForward.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "TransformerFeedForwardMoe": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for TransformerFeedForwardMoe.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "TransformerLm": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for TransformerLm.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "VisionTransformer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for VisionTransformer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
}

for layer, math in transformers.items():
    match = re.search(
        f"class {layer}\\(BaseModel\\):[\\s\\S]*?model_config = ConfigDict\\([^\\)]+\\)\\n(?:    .*\\n)*",
        content,
    )
    if match:
        block = match.group(0)
        block_clean = re.sub(r"    def __call__\(self.*?\n(?:        .*\n)*", "", block)
        content = content.replace(block, block_clean + "\n" + math.lstrip("\n"))

with open("src/zero_pax/praxis/layers/__init__.py", "w") as f:
    f.write(content)
