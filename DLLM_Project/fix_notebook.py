import json

path = "notebooks/Fase3/DLLM/XGBoost.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            new_line = line.replace('discrirminative', 'discriminative')
            if new_line != line:
                changed = True
            new_source.append(new_line)
        cell['source'] = new_source

if changed:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Corretto il typo 'discrirminative' nel notebook!")
else:
    print("Nessun typo trovato.")

