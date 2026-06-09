import re

with open("src/zero_pax/praxis/layers/__init__.py") as f:
    content = f.read()

rnns = {
    "CifgLstmCellSimple": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for CifgLstmCellSimple.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "FRnn": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for FRnn.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "LstmCellSimple": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for LstmCellSimple.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "LstmFrnn": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for LstmFrnn.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "SSM": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for SSM.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "SSMGated": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for SSMGated.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "StackFrnn": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for StackFrnn.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
    "TemporalShifting": """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        \"\"\"Forward pass for TemporalShifting.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Output array.
        \"\"\"
        return x
""",
}

for layer, math in rnns.items():
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
