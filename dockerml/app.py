import gradio as gr
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Create a directory for saving plots
os.makedirs('temp_plots', exist_ok=True)

def get_difference_color(predicted, actual):
    """
    Returns color based on prediction accuracy:
    - Green: difference < 0.03 (very accurate)
    - Yellow: difference between 0.03 and 0.05 (moderately accurate)
    - Red: difference > 0.05 (less accurate)
    """
    diff = abs(predicted - actual)
    if diff < 0.03:
        return "color: #22c55e"
    elif diff < 0.05:
        return "color: #eab308"
    else:
        return "color: #ef4444"
        
def create_styled_table(results_df):
    html = """
    <div style="background-color: #0f172a; padding: 20px; border-radius: 8px; max-width: 800px; margin: 0 auto;">
        <h2 style="color: white; font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem;">TOP 50 Tennis Player WIN% Predictions Analysis</h2>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 1px solid #334155;">
                        <th style="padding: 12px 16px; text-align: left; color: #cbd5e1; font-size: 0.875rem;">Player</th>
                        <th style="padding: 12px 16px; text-align: right; color: #cbd5e1; font-size: 0.875rem;">Predicted Win Rate</th>
                        <th style="padding: 12px 16px; text-align: right; color: #cbd5e1; font-size: 0.875rem;">Actual Win Rate</th>
                        <th style="padding: 12px 16px; text-align: right; color: #cbd5e1; font-size: 0.875rem;">Difference</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for idx, row in results_df.iterrows():
        difference = row['Actual_Win_Rate'] - row['Predicted_Win_Rate']
        diff_color = get_difference_color(row['Predicted_Win_Rate'], row['Actual_Win_Rate'])
        bg_color = "#1e293b" if idx % 2 == 0 else "#0f172a"
        
        # Make the difference value bold
        html += f"""
            <tr style="border-bottom: 1px solid #1e293b; background-color: {bg_color};">
                <td style="padding: 12px 16px; color: #e2e8f0; font-size: 0.875rem;">{row['Player']}</td>
                <td style="padding: 12px 16px; text-align: right; color: #e2e8f0; font-size: 0.875rem;">{row['Predicted_Win_Rate']:.3f}</td>
                <td style="padding: 12px 16px; text-align: right; color: #e2e8f0; font-size: 0.875rem;">{row['Actual_Win_Rate']:.3f}</td>
                <td style="padding: 12px 16px; text-align: right; font-size: 0.875rem; font-weight: bold; {diff_color}">
                    {'+' if difference > 0 else ''}{difference:.3f}
                </td>
            </tr>
        """
    
    # Add a legend for the colors
    html += """
                </tbody>
            </table>
            <div style="margin-top: 16px; font-size: 0.75rem; color: #cbd5e1;">
                <div style="margin-bottom: 4px;">
                    <span style="color: #22c55e;">■</span> Difference < 0.03 (Very accurate)
                </div>
                <div style="margin-bottom: 4px;">
                    <span style="color: #eab308;">■</span> Difference 0.03-0.05 (Moderately accurate)
                </div>
                <div>
                    <span style="color: #ef4444;">■</span> Difference > 0.05 (Less accurate)
                </div>
            </div>
        </div>
    </div>
    """
    return html
    
def load_components():
    with open('model.pkl', 'rb') as f:
        grid_search = pickle.load(f)
    with open('pipeline_full.pkl', 'rb') as f:
        pipeline_full = pickle.load(f)
    with open('top_28_features_indices.pkl', 'rb') as f:
        top_28_features_indices = pickle.load(f)
    return grid_search, pipeline_full, top_28_features_indices

def apply_feature_engineering(df):
    if 'Nationality' in df.columns:
        df.drop('Nationality', axis=1, inplace=True)
    
    df['SPW_Ace_ratio'] = df['SPW'] / df['Ace%']
    
    df['Pressure_Performance'] = df['BPConv%'] * df['TB_W%']
    df['Ace%_Tier'] = pd.qcut(df['Ace%'], q=3, labels=['Low', 'Medium', 'High'])
    df['TB_W%_Tier'] = pd.qcut(df['TB_W%'], q=3, labels=['Low', 'Medium', 'High'])
    df['Hld%_Tier'] = pd.qcut(df['Hld%'], q=3, labels=['Low', 'Medium', 'High'])
    
    for col in ['Ace%_Tier', 'TB_W%_Tier', 'Hld%_Tier']:
        df[col] = df[col].astype('category')
    
    for col in ['Aces', 'DFs', 'Points']:
        if col in df.columns:
            df[f'Log_{col}'] = np.log1p(df[col])
    
    df['BP_Efficiency'] = df['BPConv'] / df['BPChnc']
    df['BP_Efficiency'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['BP_Efficiency'].fillna(0, inplace=True)
    
    df['Serve_Quality'] = (df['SPW'] + df['Ace%'] - df['DF%'] + df['Hld%']) / 4
    df['Return_Efficiency'] = (df['RPW'] + df['Brk%'] - df['vAce%']) / 3
    df['Pressure_Efficiency'] = (df['Pressure_Performance'] + df['BPConv'] + df['BPSvd%']) / 3
    df['Break_Pressure_Interaction'] = df['Brk%'] * df['Pressure_Efficiency']
    
    if 'Time_Mt_sec' in df.columns:
        df['Log_Time_Mt_sec'] = np.log(df['Time_Mt_sec'] + 1)
    
    df['OppRk_Avg'] = (df['MdOppRk'] + df['MnOppRk']) / 2
    df['Point_Efficiency'] = df['TPW%'] / df['Sec/Pt']
    df['Break_Point_Defense'] = df['BPSvd%'] / df['BPvs/G']
    
    df['SPW_squared'] = df['SPW'] ** 2
    df['RPW_squared'] = df['RPW'] ** 2
    
    df['Serve_Return_Balance'] = df['Serve_Quality'] * df['Return_Efficiency']
    
    drop_columns = [
        'MdOppRk', 'MnOppRk', 'Pressure_Performance', 
        'Aces', 'DFs', 'Points', 'Player', 'M_serve', 'M_return', 'M', 'M_more', 
        'TBs', 'TB/S', 'Sets', 'Gms'
    ]
    
    df.drop([col for col in drop_columns if col in df.columns], axis=1, inplace=True)
    
    return df

def create_comparison_plots(actual_values, predicted_values):
    # Create Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, predicted_values, alpha=0.5)
    plt.plot([actual_values.min(), actual_values.max()], 
             [actual_values.min(), actual_values.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    comparison_plot_path = 'temp_plots/comparison.png'
    plt.savefig(comparison_plot_path)
    plt.close()

    # Create Residuals Histogram
    residuals = actual_values - predicted_values
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color='blue', alpha=0.7)
    plt.title('Residuals Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    residuals_plot_path = 'temp_plots/residuals.png'
    plt.savefig(residuals_plot_path)
    plt.close()

    return comparison_plot_path, residuals_plot_path

def predict_tennis(csv_file):
    try:
        # Load all components
        with open('model.pkl', 'rb') as f:
            grid_search = pickle.load(f)
        with open('pipeline_full.pkl', 'rb') as f:
            pipeline_full = pickle.load(f)
        with open('top_28_features_indices.pkl', 'rb') as f:
            top_28_features_indices = pickle.load(f)
        
        # Read and preprocess the CSV file
        df = pd.read_csv(csv_file.name)
        player_names = df['Player'].copy()
        
        # Save actual values if they exist
        has_actual_values = 'M_W%' in df.columns
        if has_actual_values:
            actual_values = df['M_W%'].to_numpy()
        
        # Apply initial feature engineering
        df_processed = apply_feature_engineering(df)
        
        # Apply the pipeline preprocessor
        df_preprocessed = pipeline_full.named_steps['preprocessor'].transform(df_processed)
        
        # Select top 28 features
        df_top_28 = df_preprocessed[:, top_28_features_indices]
        
        # Make predictions
        predictions = grid_search.predict(df_top_28)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Player': player_names,
            'Predicted_Win_Rate': predictions,
            'Actual_Win_Rate': actual_values if has_actual_values else predictions
        })
        
        # Create styled HTML table
        styled_table = create_styled_table(results_df)
        
        if has_actual_values:
            # Calculate metrics
            mse = mean_squared_error(actual_values, predictions)
            mae = mean_absolute_error(actual_values, predictions)
            r2 = r2_score(actual_values, predictions)
            
            metrics_text = f"""
            Model Performance Metrics:
            MSE: {mse:.4f}
            MAE: {mae:.4f}
            R² Score: {r2:.4f}
            """
            
            # Create plots
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.scatter(actual_values, predictions, alpha=0.5)
            ax1.plot([actual_values.min(), actual_values.max()], 
                    [actual_values.min(), actual_values.max()], 'k--', lw=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Actual vs Predicted Values')
            ax1.grid(True)
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            residuals = actual_values - predictions
            ax2.hist(residuals, bins=30, color='blue', alpha=0.7)
            ax2.set_title('Residuals Distribution')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
            
            return (
                styled_table,
                metrics_text,
                fig1,
                fig2
            )
        else:
            return (
                styled_table,
                "No actual values provided for comparison.",
                None,
                None
            )
            
    except Exception as e:
        return (
            f"An error occurred: {str(e)}",
            "Please ensure your CSV file has all required columns and proper formatting.",
            None,
            None
        )

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_tennis,
    inputs=gr.File(label="Upload CSV file"),
    outputs=[
        gr.HTML(label="Predictions"),
        gr.Textbox(label="Metrics", lines=5),
        gr.Plot(label="Actual vs Predicted Plot"),
        gr.Plot(label="Residuals Distribution")
    ],
    title="TOP 50 Tennis Player Current Season WINRATE Predictor",
    description="""
    Upload a CSV file containing player statistics to predict their match winning percentage.
    From the tennis-abstract.com.
    """,
    allow_flagging='never'
)

# Launch the app
if __name__ == "__main__":
    demo.launch()