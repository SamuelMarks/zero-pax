from collections import defaultdict

with open("PAX_TODO.md") as f:
    lines = f.readlines()

layers = []
for line in lines:
    if line.startswith("| [x] |") or line.startswith("| [ ] |"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 8:
            symbol = parts[4]
            layers.append(symbol)

# Categorize
categories = {
    "Activations": [
        "Activation",
        "ReLU",
        "GELU",
        "Swish",
        "SiLU",
        "Tanh",
        "Sigmoid",
        "ELU",
    ],
    "Normalizations": ["Norm"],
    "Attention": ["Attention", "PerDimScale", "RelativeBias"],
    "Convolutions": ["Conv", "Pool"],
    "Transformers": ["Transformer", "AttentionProjection", "FeedForward"],
    "RNNs & SSMs": ["Lstm", "Rnn", "SSM", "Temporal"],
    "Embeddings & Softmax": ["Embedding", "Softmax", "Ngrammer", "VectorQuantiz"],
    "Models & Architectures": [
        "Model",
        "Net",
        "Conformer",
        "BiTemperedLoss",
        "BregmanPCA",
    ],
    "Core & Base": [],
}

grouped = defaultdict(list)
for layer in layers:
    assigned = False
    for cat, keywords in categories.items():
        if any(kw.lower() in layer.lower() for kw in keywords):
            grouped[cat].append(layer)
            assigned = True
            break
    if not assigned:
        grouped["Core & Base"].append(layer)

with open("SEMANTIC_PLAN.md", "w") as f:
    f.write("# Semantic & Mathematical Implementation Plan\n\n")
    f.write(
        "This document outlines the exhaustive plan for implementing the pure-numpy mathematical semantics of the 108 Praxis layers.\n\n"
    )
    f.write("## Global Objectives\n")
    f.write(
        "- [ ] Replace dummy Pydantic fields with strict `numpy` array validation.\n"
    )
    f.write("- [ ] Implement `__call__` / `forward` methods for all modules.\n")
    f.write(
        "- [ ] Define state initialization (`init_weights`, `init_states`) purely in NumPy.\n"
    )
    f.write(
        "- [ ] Implement mathematical parity tests against expected formulas (100% test coverage).\n"
    )
    f.write("- [ ] No third-party dependencies besides `numpy` and `pydantic`.\n\n")

    for cat in sorted(grouped.keys()):
        f.write(f"## {cat}\n")
        for layer in sorted(grouped[cat]):
            f.write(f"- [ ] Implement `{layer}` semantics\n")
            f.write("  - [ ] State/Weight initialization\n")
            f.write("  - [ ] Forward pass (`__call__`) math\n")
            f.write("  - [ ] Numerical unit tests\n")
        f.write("\n")

print("Generated SEMANTIC_PLAN.md")
