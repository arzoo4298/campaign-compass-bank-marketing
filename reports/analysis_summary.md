# Campaign analysis: decision memo

## Recommendation

Use this analysis to design a controlled targeting pilot. Do not treat the model ranking as proof that extra calls cause more subscriptions.

## Dataset and quality

- 41,188 records; 4,640 subscriptions; overall historical subscription rate **11.3%**.
- The chronological holdout contains 8,238 later records with a **30.8%** subscription rate, compared with **6.4%** in training.
- The response rate shifts sharply across this time split. Treat the holdout as a drift stress test; its lift estimate may not transfer to a current campaign.
- The source includes contact duration, which is observed only after a call. It is excluded. Current-campaign call count is excluded because it changes during a campaign.
- This is historical data from 2008–2010. Channel, customer mix, prices, and response behavior may have changed.

## Model readout

- Pre-contact feature baseline: ROC-AUC **0.701**, average precision **0.505** on the most recent 20% of rows.
- Highest-scored decile conversion rate is **60.9%**; 1.97× the holdout average.
- AUC measures ranking across thresholds; average precision is useful with an imbalanced outcome. Neither measures campaign profit or causal lift.

## Segment signal

Historical subscription rates differ by previous campaign outcome:

- `success`: 65.11% (1,373 customers)
- `failure`: 14.23% (4,252 customers)
- `nonexistent`: 8.83% (35,563 customers)

Segment differences are descriptive, and small groups can have noisy rates. Review counts in `segment_rates.csv` before acting.

## Pilot design

1. Define an eligible population and contact-frequency cap with campaign owners.
2. Randomly split eligible customers: model-ranked outreach versus business-as-usual outreach (or an uncontacted holdout if ethically and operationally suitable).
3. Compare incremental subscriptions per 1,000 eligible customers and net value per contact; pre-specify guardrails and subgroup checks.
4. Expand only if the pilot shows incremental value, calibrated scores, and acceptable contact and fairness outcomes.

## Limitations

This dataset contains no contact costs, deposit value, eligibility policy, or randomized assignment. It cannot support an ROI estimate or prove that contacting a high-scoring customer causes a subscription. The chronological holdout is more realistic than a random split, but it is still from the same historical source and era.
