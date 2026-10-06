# Data dictionary and analysis assumptions

## Target

`y` is the source label (`yes` / `no`) for subscription to a term deposit. The pipeline maps this to `subscribed` (`1` / `0`).

## Main features

- Customer profile: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
- Campaign context: `contact`, `month`, `day_of_week`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- Prior contact history: `pdays`, `previous`, `poutcome`
- Excluded: `duration` (known only after contact); `campaign` (current campaign contact count can change and is not fixed at pre-contact scoring time)

The source uses `unknown` as a category for some fields and `999` for `pdays` when the client was not previously contacted. The project preserves these source meanings rather than silently treating them as ordinary missing values.

## Split

The rows are ordered by date in the source. The first 80% train the model; the latest 20% form a chronological holdout. No row shuffling is applied. This approximates scoring later campaigns, but it is not a substitute for validation on current business data.
