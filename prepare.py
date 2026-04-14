import json
import gzip
import pandas as pd

# Read the gzipped JSON file
with gzip.open('data/cvss_all_11042026.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

# Convert to CSV format with flattened CVSS metrics
rows = []
for entry in data:
    # Extract year from CVE id (format: CVE-YYYY-...)
    cve_id = entry['id']
    year = cve_id.split('-')[1] if '-' in cve_id else ''

    cvss = entry['cvss_metrics']
    rows.append({
        'id': cve_id,
        'year': year,
        'description': entry['description'],
        'attack_vector': cvss['attack_vector'],
        'attack_complexity': cvss['attack_complexity'],
        'privileges_required': cvss['privileges_required'],
        'user_interaction': cvss['user_interaction'],
        'scope': cvss['scope'],
        'confidentiality': cvss['confidentiality'],
        'integrity': cvss['integrity'],
        'availability': cvss['availability']
    })

# Create DataFrame and save as gzipped CSV
df = pd.DataFrame(rows)
df.to_csv('data/cvss_all_11042026.csv.gz', index=False, compression='gzip')

print(f'Successfully processed {len(df)} entries')
print(f'Columns: {list(df.columns)}')
df.head()
