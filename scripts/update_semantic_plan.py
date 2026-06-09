import re

with open("SEMANTIC_PLAN.md") as f:
    content = f.read()

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

for layer in transformers:
    content = content.replace(
        f"- [ ] Implement `{layer}` semantics", f"- [x] Implement `{layer}` semantics"
    )
    match = re.search(
        f"- \\[x\\] Implement `{layer}` semantics\\n  - \\[ \\] State/Weight initialization\\n  - \\[ \\] Forward pass \\(`__call__`\\) math\\n  - \\[ \\] Numerical unit tests",
        content,
    )
    if match:
        old = match.group(0)
        new = f"- [x] Implement `{layer}` semantics\n  - [x] State/Weight initialization\n  - [x] Forward pass (`__call__`) math\n  - [x] Numerical unit tests"
        content = content.replace(old, new)

with open("SEMANTIC_PLAN.md", "w") as f:
    f.write(content)
