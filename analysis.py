"""
Interactive Data Analysis Notebook
Email: 23f3003311@ds.study.iitm.ac.in
"""

import marimo as mo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CELL 1: Data Generation and Initial Setup
# ============================================================================
# This cell generates synthetic dataset and defines initial parameters
# The data flows from this cell to cells 2, 3, and 4

mo.md("# 📊 Interactive Data Analysis Dashboard")
mo.md("**Email:** 23f3003311@ds.study.iitm.ac.in")

# Generate synthetic dataset
n_samples = 200
data = {
    'Age': np.random.normal(35, 10, n_samples).clip(18, 70),
    'Income': np.random.normal(50000, 15000, n_samples).clip(20000, 120000),
    'Education_Years': np.random.normal(16, 4, n_samples).clip(10, 22),
    'Work_Experience': np.random.normal(10, 7, n_samples).clip(0, 30),
    'Satisfaction_Score': np.random.uniform(1, 10, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Add derived column: Income per Experience Year
df['Income_per_Exp'] = df['Income'] / (df['Work_Experience'] + 1)

# Display dataset info
mo.md(f"### 📋 Dataset Overview")
mo.md(f"**Sample Size:** {len(df)} records")
mo.md(f"**Variables:** {', '.join(df.columns.tolist())}")
mo.md(f"**Data Preview:**")
mo.md(df.head().to_markdown())

# ============================================================================
# CELL 2: Interactive Slider Widget for Sample Selection
# ============================================================================
# This cell creates an interactive widget that controls data sampling
# The widget state is used in Cell 3 for filtering data

mo.md("---")
mo.md("### 🎚️ Interactive Controls")

# Create slider for sample size selection
sample_slider = mo.ui.slider(
    start=50,
    stop=len(df),
    value=100,
    step=10,
    label="Select Sample Size:",
    on_change=lambda value: None  # Reactive update
)

# Create slider for correlation threshold
corr_slider = mo.ui.slider(
    start=0.1,
    stop=0.9,
    value=0.5,
    step=0.1,
    label="Correlation Threshold:",
    on_change=lambda value: None
)

# Display sliders
mo.hstack([sample_slider, corr_slider], justify="start", gap=2)

# ============================================================================
# CELL 3: Dynamic Data Analysis Based on Slider Values
# ============================================================================
# This cell depends on Cell 1 (df) and Cell 2 (slider values)
# When sliders change, this cell automatically updates

mo.md("---")
mo.md("### 📈 Analysis Results")

# Get current slider values
sample_size = sample_slider.value
corr_threshold = corr_slider.value

# Sample data based on slider
df_sampled = df.sample(n=sample_size, random_state=42)

# Calculate statistics
mean_income = df_sampled['Income'].mean()
mean_age = df_sampled['Age'].mean()
mean_satisfaction = df_sampled['Satisfaction_Score'].mean()

# Calculate correlation matrix
corr_matrix = df_sampled[['Age', 'Income', 'Education_Years', 
                          'Work_Experience', 'Satisfaction_Score']].corr()

# Find strong correlations
strong_correlations = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = abs(corr_matrix.iloc[i, j])
        if corr_value > corr_threshold:
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            strong_correlations.append((var1, var2, corr_matrix.iloc[i, j]))

# Dynamic markdown output based on widget state
mo.md(f"#### 📊 Statistics for Selected Sample ({sample_size} records)")
mo.md(f"- **Average Age:** ${mean_age:.1f}$ years")
mo.md(f"- **Average Income:** ${mean_income:,.0f}$")
mo.md(f"- **Average Satisfaction:** ${mean_satisfaction:.2f}/10$")

mo.md(f"#### 🔗 Strong Correlations (threshold: {corr_threshold})")
if strong_correlations:
    for var1, var2, corr_value in strong_correlations:
        direction = "positive" if corr_value > 0 else "negative"
        strength = "strong" if abs(corr_value) > 0.7 else "moderate"
        mo.md(f"- **{var1}** ↔ **{var2}**: ${corr_value:.3f}$ ({strength} {direction} correlation)")
else:
    mo.md("No correlations above the selected threshold.")

# ============================================================================
# CELL 4: Visualization
# ============================================================================
# This cell depends on Cell 3 (df_sampled and strong_correlations)
# Visualizations update when data changes

mo.md("---")
mo.md("### 📊 Visualizations")

# Create a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Age vs Income scatter
axes[0, 0].scatter(df_sampled['Age'], df_sampled['Income'], alpha=0.6)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Income')
axes[0, 0].set_title('Age vs Income')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Education vs Satisfaction scatter
axes[0, 1].scatter(df_sampled['Education_Years'], 
                    df_sampled['Satisfaction_Score'], 
                    alpha=0.6, color='green')
axes[0, 1].set_xlabel('Education Years')
axes[0, 1].set_ylabel('Satisfaction Score')
axes[0, 1].set_title('Education vs Satisfaction')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Correlation heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Matrix')

# Plot 4: Income distribution histogram
axes[1, 1].hist(df_sampled['Income'], bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(mean_income, color='red', linestyle='--', 
                   label=f'Mean: ${mean_income:,.0f}')
axes[1, 1].set_xlabel('Income')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Income Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
mo.mpl.interactive(fig)

# ============================================================================
# CELL 5: Data Download and Export Options
# ============================================================================
# This cell provides interactive data export functionality
# Depends on Cell 3 (df_sampled)

mo.md("---")
mo.md("### 💾 Export Options")

# Create download button for sampled data
@mo.cache
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(df_sampled)

download_button = mo.ui.download_button(
    data=csv_data,
    filename=f"sampled_data_{sample_size}_records.csv",
    label=f"📥 Download {sample_size} Records as CSV"
)

mo.md("Export the currently displayed sample:")
mo.hstack([download_button], justify="start")

# ============================================================================
# CELL 6: Summary and Insights
# ============================================================================
# Final cell that summarizes insights based on all previous cells

mo.md("---")
mo.md("### 💡 Key Insights")

# Generate insights based on current state
insights = []

if len(strong_correlations) > 0:
    insights.append(f"Found **{len(strong_correlations)}** significant correlation(s) with threshold ≥ {corr_threshold}")
else:
    insights.append("No strong correlations found with current threshold")

if mean_satisfaction > 7:
    insights.append("Overall satisfaction is relatively high (> 7/10)")
elif mean_satisfaction > 5:
    insights.append("Overall satisfaction is moderate (5-7/10)")
else:
    insights.append("Overall satisfaction needs improvement (< 5/10)")

if mean_income > 55000:
    insights.append("Sample shows above-average income levels")
else:
    insights.append("Sample shows average or below-average income levels")

mo.md("#### Summary:")
for insight in insights:
    mo.md(f"- {insight}")

mo.md("---")
mo.md("#### 🎯 Notebook Features:")
mo.md("""
1. **Reactive Cells**: Change sliders to see all dependent cells update automatically
2. **Interactive Widgets**: Sliders control sample size and correlation threshold
3. **Dynamic Visualizations**: Plots update based on selected sample
4. **Data Export**: Download filtered data as CSV
5. **Real-time Insights**: Key findings update with parameter changes
""")

mo.md("#### 📝 Data Flow Documentation:")
mo.md("""
**Cell Dependencies:**
- Cell 1 → Generates base dataset
- Cell 2 → Creates interactive sliders
- Cell 3 → Depends on Cells 1 & 2 for data analysis
- Cell 4 → Depends on Cell 3 for visualizations
- Cell 5 → Depends on Cell 3 for data export
- Cell 6 → Depends on all previous cells for insights
""")

mo.md("*Created with Marimo - Interactive Python Notebook*")
