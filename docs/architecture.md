# Architecture

## Analytical layer

1. Customer data
2. Cleaning and research feature engineering
3. Research Random Forest
4. Probability calibration
5. SHAP explanation
6. Calibrated risk segmentation
7. Revenue at Risk
8. Revenue Risk Score
9. Retention Priority Score
10. Retention targeting

## Application layer

MySQL provides persistent customer and scoring storage.

Streamlit provides a user-facing interface for:
- portfolio-level risk overview
- individual customer scoring
- retention prioritization
- established research evidence

## GenAI layer

Hugging Face is a language layer on top of already-computed model signals. It converts model outputs into plain-language business explanations. It is not the predictive engine.

## Optional Deep Learning and NLP add-on

The add-on provides two separate analysis branches. A PyTorch MLP trains on the same structured customer feature set and is compared with the calibrated Random Forest using a shared stratified hold-out split. A Transformer/NLP branch scores customer support messages and aggregates sentiment features by customer. The optional hybrid MLP can use those aggregates only when ticket data is linked to the churn cohort and each ticket predates the prediction cutoff. It does not silently replace the existing production scorer.

```text
Structured customer features -> Random Forest baseline
                            -> PyTorch MLP

Customer support messages -> Transformer sentiment -> customer aggregates
                                                    -> optional hybrid MLP

Evaluation artifacts -> Model Comparison dashboard tab
NLP customer aggregates -> Customer Voice dashboard tab
```

The included `data/demo/` files are synthetic and intended for smoke tests. The add-on integration guide documents how to run it with appropriately sourced real data.
