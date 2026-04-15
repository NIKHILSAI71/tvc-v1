with open('src/tvc/training.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if i == 1268: # Line 1269 in 1-indexed
        pass
    if i >= 1268 and i <= 1271:
        continue # delete the extra logging
    new_lines.append(line)

with open('src/tvc/training.py', 'w') as f:
    f.writelines(new_lines)
