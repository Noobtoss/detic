import re
import sys

def parse_array_ids(array_str):
    array_str = array_str.split('#')[0].strip()
    array_str = re.sub(r'%\d+', '', array_str).strip()
    ids = []
    for part in array_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return ids

def experiments_read(text, task_ids):
    exp_by_id = {}
    for line in text.splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0].isdigit():
            exp_by_id[int(parts[0])] = parts[1] if len(parts) > 1 else "?"

    experiments = []
    for task_id in task_ids:
        if task_id not in exp_by_id:
            experiments.append(f"Warning: task_id {task_id} not found in exp map, skipping.")
        else:
            experiments.append(exp_by_id[task_id])
    return experiments

if __name__ == "__main__":
    array_str = sys.argv[1] if len(sys.argv) > 1 else "1"
    map_file  = sys.argv[2] if len(sys.argv) > 2 else "experiments_map.txt"
    task_ids  = parse_array_ids(array_str)

    with open(map_file, "r") as f:
        text = f.read()

    exp_names = experiments_read(text, task_ids)
    for name in exp_names:
        print(name)
