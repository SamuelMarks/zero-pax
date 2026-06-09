import re

with open("src/zero_pax/praxis/layers/__init__.py") as f:
    content = f.read()

classes = re.findall(r"class ([a-zA-Z0-9]+)\(BaseModel\):", content)

for cls in classes:
    match = re.search(
        f"class {cls}\\(BaseModel\\):[\\s\\S]*?(?=class [a-zA-Z0-9]+\\(BaseModel\\):|$)",
        content,
    )
    if match:
        block = match.group(0)
        calls = list(
            re.finditer(r"    def __call__\(self.*?\n(?:        .*\n)*", block)
        )
        if len(calls) > 1:
            for call_match in calls:
                if "-> Any:" in call_match.group(0) or "args[0]" in call_match.group(0):
                    block_new = block.replace(call_match.group(0), "")
                    content = content.replace(block, block_new)
                    block = block_new

with open("src/zero_pax/praxis/layers/__init__.py", "w") as f:
    f.write(content)
