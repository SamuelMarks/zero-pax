import re

transformers = [
    "AdaptedTransformerFeedForward",
    "FeedForward",
    "PipelinedTransformer",
    "SSMTransformer",
    "StackedTransformer",
    "StackedTransformerRepeated",
    "Transformer",
    "TransformerEncoderDecoder",
    "TransformerFeedForward",
    "TransformerFeedForwardMoe",
    "TransformerLm",
    "VisionTransformer",
]

with open("tests/test_praxis_layers.py") as f:
    content = f.read()

for layer in transformers:
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
