The model was evaluated on the **NDBC Station 42001** dataset. Below are the performance metrics for the **Imputation Task**, where the model reconstructs missing values in the wave data.

### Performance Metrics
The model achieves low error rates across all features, demonstrating high accuracy in reconstructing missing environmental data.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **RMSE** | **0.065** | Root Mean Square Error (lower is better) |
| **MAE** | **0.042** | Mean Absolute Error (lower is better) |
| **CRPS** | **0.034** | Continuous Ranked Probability Score (captures uncertainty) |

### Visual Demonstration
The plot below demonstrates the model's ability to capture complex non-linear patterns.
* **Green Line:** The model's reconstruction.
* **Shaded Area:** The 90% confidence interval (uncertainty).
* **Blue Dots:** The ground truth targets the model successfully recovered.
  
