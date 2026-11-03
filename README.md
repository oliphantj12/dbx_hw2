# dbx_hw2

### 6. Experiment Comparison

**Setup.** Four experiments trained the same algorithm, a feed-forward neural network (MLP) with SGD, across two feature versions and two hyperparameter sets. All training data was read from the Unity Catalog feature store through `create_training_set`, and every athlete was assigned to the same train/test split in all four runs.

| Hyperparameter set | Learning rate | Epochs | Batch size | Hidden units | Dropout |
|---|---|---|---|---|---|
| hp_a | 0.05 | 20 | 250 | [64, 32] | 0.0 |
| hp_b | 0.01 | 40 | 128 | [128, 64] | 0.2 |

**Quantitative results** (test set, target = total lift in lb):

| Run | RMSE | MAE | R² |
|---|---|---|---|
| v1_hp_b | **145.6** | **112.6** | **0.725** |
| v2_hp_b | 146.1 | 113.6 | 0.722 |
| v1_hp_a | 146.2 | 112.9 | 0.722 |
| v2_hp_a | 210.9 | 165.3 | 0.422 |

- **Hyperparameters:** hp_b performed best on both feature versions. On v1 the improvement over hp_a was small (RMSE 146.2 → 145.6). On v2 it was large (210.9 → 146.1).
- **Feature versions:** under hp_b, v1 and v2 performed almost identically (RMSE 145.6 vs. 146.1). Removing athletes who declined survey questions (v2) didn't meaningfully change accuracy. That's expected, since v2 differs from v1 by about 800 rows and one feature column.
- **Instability:** v2_hp_a was a clear outlier (R² 0.42). Because v2 matches v1 under hp_b, the drop comes from training instability with hp_a's higher learning rate and short schedule, not from weaker features.

**Qualitative results** (diagnostic plots logged to MLflow for each run):
- **Loss curves:** the hp_b runs show smoother convergence, with validation loss tracking training loss closely. v2_hp_a's validation loss stays high or fluctuates, which is consistent with unstable optimization. 
- **Predicted vs. actual:** the three strong runs cluster tightly around the diagonal. All runs tend to under-predict the strongest athletes, which is typical when there are few extreme values.
- **Residuals:** roughly centered at zero for the strong runs, while v2_hp_a's residuals are wider and/or shifted, showing systematic bias. 

### 7. Carbon Emissions Comparison

Emissions were measured with CodeCarbon around model training only and logged to MLflow for each run.

| Run | Train time (s) | Energy (Wh) | CO₂ (mg) |
|---|---|---|---|
| v1_hp_a | 4.6 | 0.0084 | 3.1 |
| v2_hp_a | 4.9 | 0.0098 | 3.6 |
| v2_hp_b | 18.7 | 0.0358 | 13.2 |
| v1_hp_b | 19.0 | 0.0379 | 14.0 |

- **hp_b emitted about 3.6–4.5× more CO₂ than hp_a.** It runs twice the epochs with about half the batch size, so roughly 4× the gradient updates, and training time and emissions grew by about the same factor.
- **Accuracy vs carbon trade-off:** on v1, the extra carbon from hp_b bought only about a 0.4% reduction in RMSE. On v2, it made the difference between an unstable model and a good one. hp_b's main benefit is more reliable training, not higher peak accuracy.
- **Feature version had little effect on emissions,** since the two datasets are nearly the same size and width.
- **Caveat:** training ran on Databricks serverless compute, where CodeCarbon can't read hardware power directly. It estimates energy from runtime, an assumed CPU power and the US grid's carbon intensity (about 0.37 kg CO₂/kWh implied). The relative comparisons between runs hold up, but the absolute values are approximate. At this scale (milligrams per run) the totals are small, though in production the same ratios would apply to much larger training jobs and frequent retraining.

