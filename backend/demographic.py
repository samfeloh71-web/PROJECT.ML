import pandas as pd
import plotly.express as px
from plotly.io import to_html
import os

def calculate_demographic_trends(data):
    try:
        if data.empty:
            return None, None, None, "Uploaded CSV is empty."

        required_cols = ['CustomerID', 'OrderDate']
        optional_cols = ['Age', 'Gender', 'Income', 'Location', 'City', 'Country', 'State']
        available_cols = [col for col in required_cols + optional_cols if col in data.columns]

        if not all(col in data.columns for col in required_cols):
            return None, None, None, f"CSV must contain columns: {', '.join(required_cols)}"

        data = data.copy()
        data['OrderDate'] = pd.to_datetime(data['OrderDate'], errors='coerce')
        if data['OrderDate'].isna().all():
            return None, None, None, "Invalid OrderDate format in CSV."

        data['Month'] = data['OrderDate'].dt.to_period('M').astype(str)
        trends = {}
        visualizations = []

        trend_numeric = {}  # This will be used to construct the final table

        for col in optional_cols:
            if col not in data.columns or data[col].isnull().all():
                continue

            if col in ['Age', 'Income']:
                # ➤ Visualizations
                bar_fig = px.histogram(data, x=col, title=f'{col} Distribution (Bar Chart)')
                visualizations.append({'title': f'{col} Bar Chart', 'chart': to_html(bar_fig, full_html=False)})

                pie_counts = data[col].dropna().astype(int).astype(str).value_counts().reset_index()
                pie_counts.columns = [col, 'count']
                pie_fig = px.pie(pie_counts, names=col, values='count', title=f'{col} Distribution (Pie Chart)')
                visualizations.append({'title': f'{col} Pie Chart', 'chart': to_html(pie_fig, full_html=False)})

                # ➤ Trend (mean per month)
                trend_data = data.groupby('Month')[col].mean().reset_index()
                trend_data['Month'] = trend_data['Month'].astype(str)

                line_fig = px.line(trend_data, x='Month', y=col, title=f'{col} Trend Over Time (Mean)')
                visualizations.append({'title': f'{col} Trend Over Time', 'chart': to_html(line_fig, full_html=False)})

                trend_numeric[col] = trend_data.set_index('Month')[col]

                # For compatibility with CSV trend export
                trends[col] = {'type': 'numeric', 'trend': trend_data.set_index('Month')[col].to_dict()}

            elif col == 'Gender':
                data[col] = data[col].fillna('Unknown')
                gender_counts = data[col].value_counts().reset_index()
                gender_counts.columns = [col, 'count']

                pie_fig = px.pie(gender_counts, names=col, values='count', title=f'{col} Distribution (Pie Chart)')
                visualizations.append({'title': f'{col} Pie Chart', 'chart': to_html(pie_fig, full_html=False)})

                trends[col] = {
                    'type': 'categorical',
                    'trend': gender_counts.set_index(col)['count'].to_dict()
                }

            elif col in ['City', 'State', 'Country']:
                data[col] = data[col].fillna('Unknown')
                geo_counts = data[col].value_counts().reset_index()
                geo_counts.columns = [col, 'count']

                bar_fig = px.bar(geo_counts, x=col, y='count', title=f'{col} Distribution (Bar Chart)')
                pie_fig = px.pie(geo_counts, names=col, values='count', title=f'{col} Distribution (Pie Chart)')

                visualizations.append({'title': f'{col} Distribution', 'chart': to_html(bar_fig, full_html=False)})
                visualizations.append({'title': f'{col} Pie Chart', 'chart': to_html(pie_fig, full_html=False)})

                trends[col] = {
                    'type': 'categorical',
                    'trend': geo_counts.set_index(col)['count'].to_dict()
                }

        # ➤ Construct final combined DataFrame for numeric trends
        if trend_numeric:
            combined_df = pd.DataFrame(trend_numeric).reset_index()
            output_file = os.path.join('Uploads', 'demographic_trends_results.csv')
            combined_df.to_csv(output_file, index=False)
            combined_dict = combined_df.set_index('Month').to_dict()
            trends.update({k: {'type': 'numeric', 'trend': v} for k, v in combined_dict.items()})
        else:
            combined_df = pd.DataFrame()
            output_file = None

        return trends, output_file, visualizations, None

    except Exception as e:
        return None, None, None, f"Error in demographic trends calculation: {str(e)}"
