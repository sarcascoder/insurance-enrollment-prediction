"""Insurance enrollment prediction package.

Modules:
    config      -- central configuration (paths, feature groups, constants)
    data        -- data loading and validation
    preprocess  -- feature preprocessing pipeline (scaling + one-hot encoding)
    eda         -- exploratory data analysis, generates figures for the report
    evaluate    -- evaluation metrics and diagnostic plots
    train       -- model training, tuning, experiment tracking, model selection
    api         -- FastAPI service that serves predictions from the saved model
"""

__version__ = "1.0.0"
