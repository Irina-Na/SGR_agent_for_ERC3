import re

scores = [
    1.00,  # t01
    1.00,  # t02
    0.00,  # t03
    1.00,  # t04
    1.00,  # t05
    1.00,  # t06
    1.00,  # t07
    1.00,  # t08
    0.00,  # t09
    0.00,  # t10
    0.00,  # t11
    0.00,  # t12
    0.00,  # t13
    0.00,  # t14
    0.00,  # t15
    0.00,  # t16
    1.00,  # t17
    1.00,  # t18
    1.00,  # t19
    1.00,  # t20
    1.00,  # t21
    1.00,  # t22
    1.00,  # t23
    0.00,  # t24
    0.00,  # t25
    0.00,  # t26
    0.00,  # t27
    0.00,  # t28
    0.00,  # t29
    0.00,  # t30
    1.00,  # t31
    1.00,  # t32
    1.00,  # t33
    0.00,  # t34
    1.00,  # t35
    0.00,  # t36
    0.00,  # t37
    0.00,  # t38
    0.37,  # t39
    0.00,  # t40
    0.00,  # t41
    0.00,  # t42
    1.00,  # t43
    0.00,  # t44
    0.00,  # t45
    0.00,  # t46
    0.00,  # t47
    0.26,  # t48
    0.00,  # t49
    1.00,  # t50
    0.00,  # t51
    0.00,  # t52
    0.00,  # t53
]

log_path = r"C:\Users\Ironia\PycharmProjects\ERC3\erc3-agents\ecom-knowledge-agent\runs\20260528_235850_Qwen_Qwen3.5-397B-A17B-fast_gite6f71e6.log"

with open(log_path, "rb") as f:
    content = f.read()

# Use actual ANSI escape sequence bytes
placeholder = b'\x1b[34mScore: not available\x1b[0m'

# Find all occurrences
parts = content.split(placeholder)
print(f"Found {len(parts) - 1} score placeholders")

if len(parts) - 1 != len(scores):
    print(f"ERROR: Expected {len(scores)} scores, but found {len(parts) - 1} placeholders")
    exit(1)

# Build new content with replacements
new_parts = []
for i, part in enumerate(parts):
    new_parts.append(part)
    if i < len(scores):
        new_parts.append(f"\x1b[34mScore: {scores[i]:.2f}\x1b[0m".encode())

new_content = b"".join(new_parts)

with open(log_path, "wb") as f:
    f.write(new_content)

print("Scores added successfully!")
