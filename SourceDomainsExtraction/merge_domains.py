import csv
from pathlib import Path


def read_column_values(file_path: Path, column_name: str) -> list[str]:
    values = []
    with file_path.open(newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            value = row.get(column_name)
            if value:
                value = value.strip()
                if value:
                    values.append(value)
    return values


def merge_domains() -> None:
    script_dir = Path(__file__).resolve().parent
    fact_check_file = script_dir / 'Fact_Check' / 'top_15_org_domains.csv'
    news_file = script_dir / 'News' / 'national_sources_cpw_gt_900_20260424.csv'
    output_file = script_dir / 'merged_domains.csv'

    org_domains = read_column_values(fact_check_file, 'org_domain')
    news_domains = read_column_values(news_file, 'name')

    merged = []
    seen = set()
    for domain in org_domains + news_domains:
        if domain not in seen:
            seen.add(domain)
            merged.append(domain)

    with output_file.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['domain'])
        for domain in merged:
            writer.writerow([domain])

    print(f'Merged domains written to: {output_file}')


if __name__ == '__main__':
    merge_domains()
