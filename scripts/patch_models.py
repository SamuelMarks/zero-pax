import re

with open("src/zero_pax/praxis/layers/__init__.py") as f:
    content = f.read()

models = {
    "BertModel": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for BertModel.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "BiTemperedLoss": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for BiTemperedLoss.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "BregmanPCA": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for BregmanPCA.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "ClassificationMLPModel": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for ClassificationMLPModel.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "ClassificationModel": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for ClassificationModel.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "Conformer": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for Conformer.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "LanguageModel": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for LanguageModel.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "LanguageModelContinuousBatching": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for LanguageModelContinuousBatching.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "LanguageModelDPO": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for LanguageModelDPO.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "LanguageModelType": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for LanguageModelType.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "ResNet": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for ResNet.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "ResNetBlock": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for ResNetBlock.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "SequenceModel": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for SequenceModel.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "VanillaNet": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for VanillaNet.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
}

for layer, math in models.items():
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
