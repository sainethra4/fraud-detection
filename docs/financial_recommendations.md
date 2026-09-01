# Financial Impact and Cost-Saving Recommendations

## Purpose

The model helps investigation teams prioritise claims for review. It does not establish fraud, and it must not be used by itself to deny, reduce, or delay a legitimate claim.

## Financial impact measure

The dashboard's training report provides three AUD measures on the held-out test set:

1. **Test claim value** - total value of claims in the hold-out sample.
2. **Confirmed fraud value** - total value of claims whose historical label is fraud.
3. **Potential fraud value prioritised** - claim value of confirmed-fraud claims that the model correctly flags at its selected threshold.

The third figure is a prioritisation estimate, not a projected saving. Recovery rates, investigation cost, coverage, liability, customer remediation, and ultimately the investigator's decision determine realised savings.

## Recommendations

1. **Use a risk-ranked review queue.** Review the highest fraud-probability claims first, with claim value visible to investigators. This focuses scarce investigation time where likely exposure is higher.
2. **Set capacity-aware thresholds.** Reassess the review threshold against investigator capacity, false-positive rate, and the cost of missing fraud. Do not treat the current threshold as permanent.
3. **Strengthen upstream data quality.** Route claims with missing or inconsistent evidence, form defects, missing police reports, or missing witness information to validation workflows; missingness alone must not determine an adverse outcome.
4. **Track the investigation outcome.** Feed confirmed investigation outcomes, recovery amount, handling time, and false-positive reasons into a monthly dashboard. Retrain only after data-quality and governance review.
5. **Measure net benefit.** Track: recovered amount minus investigation cost, fraud found per investigator hour, false-positive rate, customer complaints, and claim-cycle time.
6. **Monitor model change.** Compare feature distributions, fraud rate, recall, precision, and subgroup outcomes over time. Pause or recalibrate the model if material drift or unfair impact is detected.

## Decision rule

The `investigate` flag means **manual review recommended**, not fraud confirmed. An authorised investigator should document independent evidence and the final outcome for every reviewed claim.

