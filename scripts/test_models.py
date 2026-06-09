import re

models = [
    "BertModel",
    "BiTemperedLoss",
    "BregmanPCA",
    "ClassificationMLPModel",
    "ClassificationModel",
    "Conformer",
    "LanguageModel",
    "LanguageModelContinuousBatching",
    "LanguageModelDPO",
    "LanguageModelType",
    "ResNet",
    "ResNetBlock",
    "SequenceModel",
    "VanillaNet",
]

with open("tests/test_praxis_layers.py") as f:
    content = f.read()

for layer in models:
    old_test = f"""def test_{layer.lower()}():
    \"\"\"Test {layer}.\"\"\"
    obj = {layer}()
    assert obj is not None
    import numpy as np
    out = obj(np.random.randn(2, 4))
    assert out is not None"""

    new_test = f"""def test_{layer.lower()}():
    \"\"\"Test {layer}.\"\"\"
    import numpy as np
    obj = {layer}()
    assert obj is not None
    x = np.random.randn(2, 4)
    out = obj(x)
    assert out.shape == x.shape"""

    content = re.sub(
        f"def test_{layer.lower()}\\(\\):\\n(?:    .*\\n)+?(?=def test_|$)",
        new_test + "\n\n\n",
        content,
    )

with open("tests/test_praxis_layers.py", "w") as f:
    f.write(content)
