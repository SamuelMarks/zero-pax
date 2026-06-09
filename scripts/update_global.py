with open("SEMANTIC_PLAN.md") as f:
    content = f.read()

content = content.replace(
    "- [ ] Replace dummy Pydantic fields with strict `numpy` array validation.",
    "- [x] Replace dummy Pydantic fields with strict `numpy` array validation.",
)
content = content.replace(
    "- [ ] Implement `__call__` / `forward` methods for all modules.",
    "- [x] Implement `__call__` / `forward` methods for all modules.",
)
content = content.replace(
    "- [ ] Define state initialization (`init_weights`, `init_states`) purely in NumPy.",
    "- [x] Define state initialization (`init_weights`, `init_states`) purely in NumPy.",
)
content = content.replace(
    "- [ ] Implement mathematical parity tests against expected formulas (100% test coverage).",
    "- [x] Implement mathematical parity tests against expected formulas (100% test coverage).",
)
content = content.replace(
    "- [ ] No third-party dependencies besides `numpy` and `pydantic`.",
    "- [x] No third-party dependencies besides `numpy` and `pydantic`.",
)

with open("SEMANTIC_PLAN.md", "w") as f:
    f.write(content)
