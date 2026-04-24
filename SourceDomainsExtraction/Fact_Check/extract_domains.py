import pandas as pd
from urllib.parse import urlparse


INPUT_FILE = "SourceDomainsExtraction/Fact_Check/fact_check_insights/claim_review.csv"
OUTPUT_ORG = "SourceDomainsExtraction/Fact_Check/org_domains.csv"
OUTPUT_ARTICLE = "SourceDomainsExtraction/Fact_Check/article_domains.csv"


def extract_domain(url):
    if pd.isna(url):
        return None
    try:
        domain = urlparse(str(url)).netloc.lower()
        return domain.replace("www.", "")
    except Exception:
        return None


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")

    # Extract domains
    print("Extracting domains...")
    df["org_domain"] = df["author.url"].apply(extract_domain)
    df["article_domain"] = df["url"].apply(extract_domain)

    # Deduplicate
    org_domains = df["org_domain"].dropna().drop_duplicates().sort_values()
    article_domains = df["article_domain"].dropna().drop_duplicates().sort_values()

    # Save results
    org_domains.to_csv(OUTPUT_ORG, index=False)
    article_domains.to_csv(OUTPUT_ARTICLE, index=False)

    print("\nDone ✅")
    print(f"Unique fact-checking org domains: {len(org_domains)}")
    print(f"Unique article domains: {len(article_domains)}")

    print(f"\nSaved:")
    print(f" - {OUTPUT_ORG}")
    print(f" - {OUTPUT_ARTICLE}")

    # Show top orgs
    print("\nTop 10 domains:")
    print(df["org_domain"].value_counts().head(10))

    top_15 = df["org_domain"].value_counts().head(15)
    top_15.to_csv("SourceDomainsExtraction/Fact_Check/top_15_org_domains.csv")



if __name__ == "__main__":
    main()
