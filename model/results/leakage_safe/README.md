# Leakage-safe experiment record

## Protocol

* Batches 1--70: fit MinMaxScaler and train the existing GRU-AE.
* Batches 71--80: normal-only threshold calibration.  There are zero positive
  validation windows; no F1/threshold optimisation against labels is possible.
* Batches 81--100: one held-out evaluation.  A positive decision means that
  the fault flag is active at the end of the 60-sample causal window.
* Sensor scales are median/MAD estimates from training reconstruction errors.
  Equal sensor aggregation and max aggregation are fixed rules; no test-label
  sensor ranking is used.

## Recommended operational point

`robust_max_val_max` alarms when the maximum standardized end-time sensor
error exceeds the largest normal validation score.  It is deliberately more
conservative than the 99.5th-percentile alternative.

| Measure | Result |
| --- | ---: |
| ROC-AUC (threshold-independent) | 0.9682 |
| Average precision | 0.7299 |
| Precision | 0.8366 |
| Recall | 0.4776 |
| F1 | 0.6081 |
| Test-normal window FPR | 0.0061 |
| Faulty-batch event recall | 10/10 |
| Normal test batches with an alarm | 1/10 |

The 99.5th-percentile policy has F1 0.6130 and recall 0.5970, but its
test-normal FPR is 0.0230 and it alarms in 2/10 normal test batches.  It is a
useful sensitivity point, not the recommended deployment threshold.

## Important limitation

All faulty episodes are in the final test partition.  The result is a
hold-out measurement, but no fault-containing development data remains for
tuning persistence, sensor choices, or alarm delay.  In particular, three
events are first detected 180 time units after fault onset under the
conservative policy.  Obtain or reserve a separate fault-containing
development partition before making any further performance-driven changes.

## Files

* `metrics.csv`: every predeclared GRU score and normal-only calibration
  policy.
* `pca_baseline_metrics.csv`: normal-only PCA baselines under the same split.
* `primary_event_details.csv`: per-batch alarm/delay details for the
  conservative policy.
* `sensor_diagnostics_test_descriptive_only.csv`: descriptive error summary;
  it must not be used to retroactively select sensor weights.
