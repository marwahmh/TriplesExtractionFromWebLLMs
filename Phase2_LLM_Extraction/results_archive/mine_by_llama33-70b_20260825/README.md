MINE judgments produced by llama33-70b (2026-08-25), copied before re-judging everything with
gpt-oss-120b.

Kept for two reasons. The judgment logs are untagged by judge, so the gpt-oss run overwrites them
in place. And llama's verdicts on the arms both judges cover are a cross-judge comparison worth
reporting: MINE scores are known to swing widely with the judge (86.0% vs 62.7% on identical gemini
validation graphs), and that spread is the measurement's own error bar.

Coverage here is partial (591 of 720 scorings). llama was abandoned because generation on the
shared A6000 pair fell to ~0.9 tok/s under its own load, which put the remainder at ~21 hours.
