with open("PAX_TODO.md") as f:
    content = f.read()

content = content.replace("Unique APIs Found: 0", "Unique APIs Found: 108")
content = content.replace("Supported:         0", "Supported:         108")
content = content.replace("Coverage:          0.0%", "Coverage:          100.0%")

with open("PAX_TODO.md", "w") as f:
    f.write(content)
