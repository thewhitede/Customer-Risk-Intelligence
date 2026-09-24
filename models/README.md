# Model Artifacts

The original research notebook established the model architecture and held-out research metrics. A serialized binary was not part of the current project files, so this folder deliberately does not fabricate one.

Generate a deployment-oriented refit with:

```bash
python scripts/train_production_model.py
```

This creates:

- `calibrated_research_rf.joblib`
- `model_metadata.json`

The generated model is a production refit using the same research architecture. It is not a replacement for the established held-out research evaluation.
