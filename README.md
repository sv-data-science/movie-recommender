# Movie Recommender — Autoencoder on MovieLens 1M

Final project for CSCI S-89 (Deep Learning), Harvard Extension School.

An autoencoder-based collaborative filtering recommender trained on the
MovieLens 1M dataset (1,000,209 ratings, 6,040 users, 3,706 movies). Five
configurations were compared; Model 4 (early stopping + smaller bottleneck +
L2 regularization) was selected.

## Contents

- `Movie_Recommender_CSCI_S89_Final_Project.ipynb` — the project notebook
  (outputs stripped). Includes a corrected evaluation section at the end.
- `masked_rmse_fix.py` — standalone drop-in fix (same code as the notebook's
  final section).

## Evaluation fix (masked RMSE)

The original notebook trained and validated with a plain RMSE over all 3,706
entries per user. Since the validation matrix holds exactly one real rating
per user (the rest are zeros for unrated movies), the reported validation
RMSE (~0.10) was diluted ~3,706x and did not measure rating-prediction
accuracy. The corrected code scores **only actually-rated entries**
(ratings are 1–5, never 0, so nonzero == rated):

- `masked_rmse` / `masked_mae` — use as the Keras loss/metric for honest
  training and validation.
- `honest_metrics(model, X_in, held_out)` — re-scores already-trained models
  on the held-out ratings with no retraining; rebuilds the identical
  leave-one-out split (seed 42).

With proper masking, expect validation RMSE in the credible ~0.85–1.0 range
for this setup rather than ~0.10.

## Data

MovieLens 1M (`ratings.dat`, `movies.dat`) is not committed here; download it
from https://grouplens.org/datasets/movielens/1m/ and place the `.dat` files
next to the notebook before running.
