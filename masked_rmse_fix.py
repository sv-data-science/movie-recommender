"""
Corrected evaluation for the Movie Recommender autoencoder (CSCI S-89).

THE BUG
-------
The notebook trains with fit(X_train, X_train, validation_data=(X_val, X_val))
using a plain RMSE over all 3,706 entries per user. But X_val contains exactly
ONE real rating per user -- the other ~3,705 entries are zeros for unrated
movies, which the model trivially predicts as ~0. The single real rating's
error is therefore diluted ~3,706x, which is why the reported validation RMSE
(~0.0956) is ~9x better than the best published MovieLens 1M result (~0.85)
and why validation error sits *below* training error. The metric measures
"how well the model outputs zeros", not rating-prediction accuracy.

THE FIX
-------
Score ONLY actually-rated entries (ratings are 1-5, never 0, so nonzero ==
rated). Two drop-in pieces below:

  Part A -- masked loss/metric for (re)training. Makes both the training
            loss and the validation metric honest. Early stopping on
            val_masked_rmse then actually means something.

  Part B -- honest re-evaluation of your ALREADY-TRAINED models with NO
            retraining. Rebuilds the identical leave-one-out split
            (same seed=42, same loop order) and scores each model only on
            the held-out ratings. Run this in your Jupyter env where the
            trained models and data live.

Paste each part into its own notebook cell, in order.
"""

# ======================================================================
# Part A -- masked RMSE / MAE for training (drop-in replacements)
# ======================================================================
import tensorflow as tf


def masked_rmse(y_true, y_pred):
    """RMSE computed only over actually-rated entries (nonzero y_true)."""
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    sq_err = tf.square((y_true - y_pred) * mask)
    denom = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.sqrt(tf.reduce_sum(sq_err) / denom)


def masked_mae(y_true, y_pred):
    """MAE computed only over actually-rated entries (nonzero y_true)."""
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)
    abs_err = tf.abs((y_true - y_pred) * mask)
    denom = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(abs_err) / denom


# Use it when compiling (replaces loss=rmse, metrics=["mae"]):
#   autoencoder.compile(loss=masked_rmse,
#                       optimizer=optimizers.Adam(learning_rate=0.01),
#                       metrics=[masked_mae])
# And point EarlyStopping at the honest metric:
#   early_stop = callbacks.EarlyStopping(monitor="val_masked_rmse",
#                                        patience=3, restore_best_weights=True)


# ======================================================================
# Part B -- honest re-evaluation of trained models (NO retraining)
# ======================================================================
# Run this AFTER the cells that build `user_item_matrix` (it only needs
# the matrix and a trained model object, e.g. autoencoder_v4).
import numpy as np


def rebuild_leave_one_out(user_item_matrix, seed=42):
    """Rebuild the notebook's exact split: fresh RNG, same order -> identical."""
    rng = np.random.default_rng(seed=seed)
    held_out = []  # (user_idx, movie_idx, true_rating)
    X_in = user_item_matrix.copy()
    for user in range(user_item_matrix.shape[0]):
        rated = np.where(user_item_matrix[user] > 0)[0]
        if len(rated) > 1:
            chosen = rng.choice(rated)
            held_out.append((user, int(chosen), float(user_item_matrix[user, chosen])))
            X_in[user, chosen] = 0.0
    return X_in, held_out


def honest_metrics(model, X_in, held_out, batch_size=256):
    """RMSE/MAE scored ONLY on the held-out ratings. No retraining."""
    preds = model.predict(X_in, batch_size=batch_size, verbose=0)
    se, ae = 0.0, 0.0
    for u, m, r in held_out:
        err = r - float(preds[u, m])
        se += err * err
        ae += abs(err)
    n = len(held_out)
    return float(np.sqrt(se / n)), float(ae / n)


# ---- usage: score all five trained models on the honest metric ----
# X_in, held_out = rebuild_leave_one_out(user_item_matrix)
# print(f"Held-out ratings: {len(held_out)}")
# for name, model in [("Model 1", autoencoder_v1),
#                     ("Model 2", autoencoder_v2),
#                     ("Model 3", autoencoder_v3),
#                     ("Model 4", autoencoder_v4),
#                     ("Model 5", autoencoder_v5)]:
#     rmse, mae = honest_metrics(model, X_in, held_out)
#     print(f"{name}: honest val RMSE={rmse:.4f}  MAE={mae:.4f}")
#
# EXPECT: RMSE around 0.85-1.0 (the credible range for this setup on
# MovieLens 1M), NOT ~0.10. The model ranking may change -- that is the
# honest comparison to put on the portfolio.
