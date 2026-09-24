# Customer Risk Intelligence: An Explainable and Revenue-Aware Framework for Telecom Churn Prioritization

**Author:** Sudeen Koyande
**Status:** Upgraded research manuscript draft | September 2026

## Abstract

Customer churn prediction is often evaluated primarily through classification performance, yet operational retention requires more than a binary label. This study develops an explainable and revenue-aware customer risk intelligence framework that links churn prediction, probability calibration, SHAP explanation, calibrated risk segmentation, expected monthly revenue-at-risk, and retention prioritization. The research model is a Random Forest trained on the Telco Customer Churn dataset with business-oriented features for annualized revenue, tenure bands, and monthly-charge bands. On a held-out test set, the research Random Forest achieved 0.7686 accuracy, 0.5548 precision, 0.6497 recall, 0.5985 F1, 0.8208 ROC-AUC, and 0.6069 PR-AUC. Sigmoid calibration reduced Brier score from 0.1562 to 0.1431 and log loss from 0.4768 to 0.4397, while calibrated ROC-AUC was 0.8276. Risk segmentation produced observed churn rates of 11.78%, 44.63%, and 69.73% for Low, Medium, and High risk, respectively. Expected monthly revenue exposure was $27,257.11 across the held-out evaluation population. A 60/40 churn-probability/revenue-risk priority score produced retention tiers and a resource-allocation framework. Sensitivity analysis across four tested weight configurations yielded similar churn capture while changing financial exposure per targeted customer. The contribution is an integrated decision-support framework rather than a novel predictive algorithm. Results are interpreted as predictive and operational evidence, not as causal estimates of prevented churn.

**Keywords:** customer churn; telecom analytics; Random Forest; probability calibration; SHAP; explainable AI; revenue at risk; retention prioritization

## 1. Introduction

Customer churn is a recurring problem in subscription businesses because customer departure can reduce recurring revenue and increase the resources required to replace lost customers. Machine-learning models can identify customers with elevated churn likelihood, but a class label alone does not tell a retention team how reliable the probability is, why the model is concerned, how customers should be segmented operationally, or how much financial exposure is associated with a predicted churn event.

This study frames telecom churn as a customer-risk intelligence problem rather than a classification-only problem. The central research question is: **Can an explainable machine-learning framework identify high-risk telecom customers and prioritize retention interventions using churn probability and customer value?**

The framework is deliberately modular. A research Random Forest produces churn probabilities; post-hoc calibration improves the probability scale; SHAP provides global and local explanations; calibrated probabilities define risk bands; revenue-at-risk introduces customer value; and a retention-priority score ranks intervention candidates. The contribution is therefore an integrated, decision-centric framework assembled from established methods, not a claim of a new learning algorithm.

## 2. Related Work

Telecom churn research spans logistic and other statistical models, decision trees and ensembles, support vector methods, boosting algorithms, and more recent deep-learning approaches. A recent systematic review of peer-reviewed telecom churn studies reports substantial methodological diversity across both benchmark and operator datasets [1]. This body of work establishes churn prediction as a mature applied machine-learning task, but it also shows that performance comparisons alone do not fully address deployment and retention decision needs.

Explainable AI has increasingly become part of churn analytics. SHAP provides an additive feature-attribution framework that can be used to explain individual model outputs and summarize global feature importance [2]. Recent telecom studies have explicitly paired churn prediction with SHAP-style explanations to make model outputs more interpretable for decision support [3,4].

Probability quality is a different property from ranking discrimination. Niculescu-Mizil and Caruana demonstrated that classifiers can produce probabilities that are systematically distorted and that post-hoc calibration can improve their usefulness [5]. This is directly relevant when predicted probabilities are subsequently multiplied by customer value to estimate expected exposure.

Class imbalance further motivates careful metric selection. Precision-recall analysis can be more informative than ROC analysis when the positive class is relatively uncommon [6]. Accordingly, this study reports PR-AUC for the research model and supplements standard predictive metrics with risk-concentration and targeting analyses.

## 3. Dataset and Data Preparation

The study uses the Telco Customer Churn benchmark distributed as an IBM Cognos sample and widely reused in churn analytics [7]. The dataset contains 7,043 customers and 21 original variables covering demographics, service subscriptions, contract characteristics, payment methods, and billing information.

Data preparation included numeric conversion of `TotalCharges`, missing-value handling within the modeling pipeline, duplicate-row checks, duplicate-customer checks, and construction of three business-oriented research features. The audit found zero duplicate rows and zero duplicate customer identifiers; 11 `TotalCharges` entries were blank before numeric conversion.

### 3.1 Research feature engineering

`AnnualRevenue = MonthlyCharges x 12`. `TenureGroup` uses five bands (0-6, 7-12, 13-24, 25-48, 49-72 months). `MonthlyChargeGroup` uses four bands (Low, Medium-Low, Medium-High, High).

## 4. Experimental Design

The research experiment uses a stratified 80/20 train-test split with `random_state=42`, yielding 5,634 training observations and 1,409 held-out evaluation observations. The research preprocessing pipeline uses median imputation and standardization for numerical variables and most-frequent imputation followed by one-hot encoding for categorical variables. The Random Forest uses 300 trees, balanced class weights, `random_state=42`, and parallel execution across available CPU cores [8].

The held-out set is reserved for final model evaluation, calibration assessment, risk validation, and downstream targeting evaluation.

## 5. Baseline and Research Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 | - |
| Baseline Random Forest | 0.7658 | 0.5521 | 0.6230 | 0.5854 | 0.8218 | - |
| Research Random Forest | 0.7686 | 0.5548 | 0.6497 | 0.5985 | 0.8208 | 0.6069 |

The research Random Forest improves recall relative to the baseline Random Forest while achieving a similar ROC-AUC. Logistic Regression retains the highest ROC-AUC of the reported models. The research model is therefore not presented as universally superior; it is the probability-generating model for the integrated downstream framework.

## 6. Probability Calibration

| Metric | Original | Calibrated | Change |
|---|---:|---:|---:|
| Brier score | 0.1562 | 0.1431 | -0.0131 |
| ROC-AUC | 0.8208 | 0.8276 | +0.0068 |
| Log loss | 0.4768 | 0.4397 | -0.0371 |

Calibration is performed with sigmoid scaling using five-fold cross-validation on the training data while the held-out test set remains reserved for final evaluation [5,8].

## 7. Explainability with SHAP

SHAP is used at global and individual levels. Global mean absolute SHAP values identify influential features including month-to-month contract status, tenure, fiber-optic service, `TotalCharges`, two-year contract status, absence of online security, absence of technical support, electronic-check payment method, and `MonthlyCharges` [2]. Individual explanations are represented using SHAP waterfall plots.

SHAP is interpreted as a model-explanation mechanism, not a causal inference tool.

## 8. Calibrated Risk Segmentation

| Risk | Customers | Customer share | Observed churn |
|---|---:|---:|---:|
| Low | 917 | 65.08% | 11.78% |
| Medium | 307 | 21.79% | 44.63% |
| High | 185 | 13.13% | 69.73% |

Operational bands are Low < 0.30, Medium = 0.30 to < 0.60, and High >= 0.60. Observed churn rises monotonically across the groups.

## 9. Revenue-at-Risk Modeling

Expected monthly revenue at risk is defined as:

`RevenueAtRisk = CalibratedChurnProbability x MonthlyCharges`

Total expected monthly revenue exposure across the held-out population is **$27,257.11**. This is expected exposure, not guaranteed lost revenue.

## 10. Revenue-Aware Retention Prioritization

`RetentionPriorityScore = 0.60 x CalibratedChurnProbability + 0.40 x RevenueRiskScore`

| Priority | Customers | Churners | Observed churn | Revenue at risk |
|---|---:|---:|---:|---:|
| Low | 870 | 93 | 10.69% | $5,691.25 |
| Moderate | 280 | 119 | 42.50% | $7,573.33 |
| High | 187 | 107 | 57.22% | $9,115.72 |
| Critical | 72 | 55 | 76.39% | $4,876.81 |

The 60/40 configuration is the default project configuration, not a universal optimum.

## 11. Retention Targeting Efficiency

| Targeted group | Customers | Customer share | Churner capture | Efficiency |
|---|---:|---:|---:|---:|
| High | 185 | 13.13% | 34.49% | 2.63 |
| High + Medium | 492 | 34.92% | 71.12% | 2.04 |
| All | 1,409 | 100.00% | 100.00% | 1.00 |

The targeting results illustrate a coverage-versus-focus trade-off. They do not establish causal intervention effectiveness.

## 12. Retention Weight Sensitivity

| Weights | Customers targeted | Churners captured | Churn capture | Revenue risk/customer | Revenue at risk targeted |
|---|---:|---:|---:|---:|---:|
| 50/50 | 141 | 103 | 27.54% | $62.90 | $8,869.52 |
| 60/40 | 141 | 102 | 27.27% | $62.57 | $8,822.55 |
| 70/30 | 141 | 103 | 27.54% | $62.18 | $8,766.83 |
| 80/20 | 141 | 103 | 27.54% | $61.19 | $8,627.75 |

## 13. Discussion

The main empirical result is not a single best classifier. The findings support a staged transformation of model output into a decision-support artifact: prediction, calibration, explanation, segmentation, financial exposure, and retention prioritization. High-risk customers represent 13.13% of the held-out population yet contain 34.49% of observed churners. Adding Medium-risk customers raises coverage to 34.92% and captures 71.12% of churners.

The revenue-aware layer adds customer-value information without claiming a complete customer-lifetime-value model.

## 14. Threats to Validity and Limitations

The main limitations are the static benchmark dataset, single held-out split, operational rather than cost-optimized risk thresholds, non-causal retention ranking, simplified revenue exposure, non-causal SHAP interpretation, and the need for production monitoring. Future studies should add repeated/external validation, confidence intervals, threshold optimization under explicit intervention costs, causal retention-effect estimation, richer customer-value measures, and live feedback.

## 15. Reproducibility and Research Artifacts

The structured project repository contains the analytical notebooks, research-results workbook, production-oriented modules, SQL artifacts, tests, application code, and documentation. The analytical path remains organized across notebooks 01-04, while the master-results notebook and workbook are consolidation artifacts.

## 16. Conclusion

This study developed an explainable and revenue-aware customer risk intelligence framework for telecom churn. The work supports a transition from model-centric churn prediction toward decision-centric customer risk management by integrating probability quality, explainability, risk concentration, expected financial exposure, and retention prioritization.

## References

[1] Telecom customer churn prediction in telecom: a detailed systematic review of data, methodology, and evaluation methods. Telecommunication Systems, 2026.

[2] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30, 4765-4774.

[3] Zerine, I., et al. (2026). Explainable churn prediction in telecom with tabular ML five model benchmark and SHAP analysis. Discover Artificial Intelligence, 6, 263.

[4] Gaddamwar, S. (2026). Explainable Artificial Intelligence for Customer Churn Prediction in Telecommunications: A Framework for CRM Integration. arXiv:2608.26151.

[5] Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. Proceedings of ICML 2005, 625-632. doi:10.1145/1102351.1102430.

[6] Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE, 10(3), e0118432.

[7] IBM Cognos Analytics. Telco Customer Churn sample dataset.

[8] Project research artifacts: 03_research_model_calibration.ipynb and 04_risk_revenue_retention.ipynb.
