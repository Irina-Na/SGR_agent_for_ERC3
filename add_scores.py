import re

scores = {
    't01': 1.00, 't02': 1.00, 't03': 1.00, 't04': 1.00, 't05': 1.00, 't06': 0.00,
    't07': 1.00, 't08': 1.00, 't09': 0.00, 't10': 0.00, 't11': 0.00, 't12': 0.00,
    't13': 0.00, 't14': 0.00, 't15': 0.00, 't16': 0.00, 't17': 1.00, 't18': 1.00,
    't19': 1.00, 't20': 0.60, 't21': 1.00, 't22': 1.00, 't23': 0.00, 't24': 0.00,
    't25': 0.00, 't26': 0.00, 't27': 1.00, 't28': 0.00, 't29': 0.00, 't30': 0.00,
    't31': 1.00, 't32': 1.00, 't33': 1.00, 't34': 0.00, 't35': 1.00, 't36': 0.00,
    't37': 1.00, 't38': 0.00, 't39': 0.07, 't40': 0.00, 't41': 0.00, 't42': 0.00,
    't43': 1.00, 't44': 1.00, 't45': 0.00, 't46': 0.00, 't47': 0.00, 't48': 0.00,
    't49': 0.00, 't50': 0.00, 't51': 0.00,
}

filepath = r'ecom-knowledge-agent\runs\20260528_182854_Qwen_Qwen3.5-397B-A17B-fast_score_na_gitadccbff.log'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

task_idx = 0
new_lines = []
for line in lines:
    # strip ANSI codes and whitespace, then check
    clean = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
    if clean == 'Score: not available':
        task_num = f"{task_idx + 1:02d}"
        task_key = f't{task_num}'
        # preserve original line structure, just swap the value
        new_line = line.replace('not available', str(scores[task_key]))
        new_lines.append(new_line)
        task_idx += 1
    else:
        new_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Done. Tasks updated: {task_idx}")
