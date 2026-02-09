import os
from dotenv import load_dotenv

# ✅ Set up backend directory path FIRST (before using it)
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

# 1. Load the key from your .env file
load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GOOGLE_API_KEY")

# Load optional secret env file (gitignored)
secret_env = os.path.join(backend_dir, '.env.secret')
if os.path.exists(secret_env):
    load_dotenv(secret_env)

from langchain_google_genai import ChatGoogleGenerativeAI

# 2. Initialize the Gemini "Brain" with Customer Care Persona
# Temperature 0.45: Warm, human-like responses (not robotic at 0, not too random at 1)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=api_key,
    temperature=0.45
)

from flask import Flask, render_template, request, Response, jsonify, session, send_file, url_for, send_from_directory, redirect
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
 
def role_required(*roles):
    """Decorator to require the current session role to be in `roles`.

    If no user in session -> redirect to login. If user role not permitted -> render error page.
    """
    def wrapper(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user' not in session:
                return redirect(url_for('login'))
            if roles and session.get('role') not in roles:
                return render_template('error.html', error='Access Denied', message="You don't have permission to access this page.")
            return f(*args, **kwargs)
        return decorated_function
    return wrapper
import time
import threading
import pandas as pd
from io import StringIO
import os
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# ✅ For charting
import plotly.express as px
from plotly.io import to_html

# ✅ LangChain DataFrame Agent for natural language data queries
# Note: Using SimpleMemory instead of experimental agents to avoid dependency issues
# from langchain_experimental.agents import create_pandas_dataframe_agent

# ✅ LangChain Memory for conversation context (Customer Care)
# Simple conversation memory implementation
class SimpleMemory:
    """Simple conversation memory to store chat history"""
    def __init__(self, memory_key="chat_history"):
        self.memory_key = memory_key
        self.messages = []
    
    def save_context(self, inputs, outputs):
        """Save user input and AI output to memory"""
        if isinstance(inputs, dict):
            user_msg = inputs.get("input", str(inputs))
        else:
            user_msg = str(inputs)
        
        if isinstance(outputs, dict):
            ai_msg = outputs.get("output", str(outputs))
        else:
            ai_msg = str(outputs)
        
        self.messages.append({"user": user_msg, "ai": ai_msg})
    
    def load_memory_variables(self, inputs=None):
        """Load memory as formatted string"""
        chat_history = ""
        for msg in self.messages[-5:]:  # Keep last 5 exchanges
            if msg["user"]:
                chat_history += f"Customer: {msg['user']}\n"
            if msg["ai"]:
                chat_history += f"Assistant: {msg['ai']}\n\n"
        return {self.memory_key: chat_history}

# ✅ Your own module
from demographic import calculate_demographic_trends

# ✅ Set up paths for frontend templates and static files
template_folder = os.path.join(project_root, 'frontend', 'templates')
static_folder = os.path.join(project_root, 'frontend', 'static')

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ✅ SECURITY CONFIGURATION - AI Safety Guardrails
DANGEROUS_KEYWORDS = [
    'delete', 'drop', 'truncate', 'alter', 'update', 'insert',
    'modify', 'remove', 'drop table', 'execute', 'exec',
    'import os', 'import sys', '__import__', 'eval', 'exec',
    'drop column', 'flush', 'purge', 'destroy', 'erase',
    'sys.', 'os.', 'subprocess', 'shell', 'lambda',
    'exec(', 'eval(', '__code__', 'globals()', 'locals()'
]

ANALYTICS_KEYWORDS = [
    'average', 'mean', 'median', 'sum', 'total', 'count', 'min', 'max',
    'trend', 'analyze', 'analysis', 'insight', 'pattern', 'distribution',
    'correlation', 'relationship', 'segment', 'group', 'filter', 'sort',
    'customer', 'data', 'metric', 'statistic', 'churn', 'revenue', 'spending',
    'income', 'age', 'purchase', 'order', 'sales', 'region', 'category',
    'how many', 'how much', 'what is', 'show me', 'tell me', 'list', 'find',
    'top', 'best', 'worst', 'highest', 'lowest', 'compare', 'difference'
]

GREETING_KEYWORDS = ['hello', 'hi', 'help', 'how', 'what', 'feature', 'platform', 'use']

# ✅ VISUALIZATION CONFIGURATION - Chart generation keywords
VISUALIZATION_KEYWORDS = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'show me', 'display', 'diagram']
VISUALIZATION_TYPES = {
    'pie': ['pie', 'distribution', 'breakdown', 'percentage', 'share'],
    'bar': ['bar', 'comparison', 'across', 'by', 'per'],
    'line': ['trend', 'over time', 'trend line', 'line chart', 'time series'],
    'scatter': ['scatter', 'correlation', 'relationship', 'scatter plot'],
    'histogram': ['histogram', 'frequency', 'histogram distribution'],
}

# ✅ LOCALIZATION CONFIGURATION - Common region/location keywords
COMMON_REGIONS = {
    'nairobi': ['nairobi', 'nbi', 'cbd'],
    'mombasa': ['mombasa', 'msa', 'coastal'],
    'kisumu': ['kisumu', 'western'],
    'nakuru': ['nakuru', 'rift valley'],
    'kampala': ['kampala', 'uganda'],
    'dar es salaam': ['dar es salaam', 'dar', 'tanzania'],
    'accra': ['accra', 'ghana'],
    'lagos': ['lagos', 'nigeria'],
    'johannesburg': ['johannesburg', 'joburg', 'south africa'],
}

NON_TECHNICAL_TERMS = {
    'average': 'typical',
    'median': 'middle',
    'mean': 'average',
    'sum': 'total',
    'count': 'number of',
    'distribution': 'spread across',
    'correlation': 'connection between',
    'segment': 'group of',
    'cohort': 'group of',
    'metric': 'measurement',
    'outlier': 'unusual data point',
}

# Load configuration from environment variables
app.secret_key = os.getenv('SECRET_KEY', 'your-super-secret-key-change-this-in-production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', os.path.join(backend_dir, 'uploads'))
app.config['OUTPUT_FOLDER'] = os.getenv('OUTPUT_FOLDER', os.path.join(backend_dir, 'outputs'))
app.config['USERS_FILE'] = os.getenv('USERS_FILE', os.path.join(backend_dir, 'users.json'))
app.config['ALLOW_DEV_LOGIN'] = os.getenv('ALLOW_DEV_LOGIN', 'false').lower() == 'true'
app.config.setdefault('USERS', None)  # will be loaded on demand
uploaded_data = {}  # Store uploaded DataFrame by file_id

# --- MongoDB setup (pymongo) ---
MONGODB_URI = os.getenv('MONGODB_URI')
mongo_client = None
mongo_db = None
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        mongo_client = MongoClient(MONGODB_URI)
        # default database from URI if present, otherwise use 'app_db'
        mongo_db = mongo_client.get_default_database() or mongo_client['app_db']
        print('Connected to MongoDB')
    except Exception as e:
        print(f'Warning: Could not connect to MongoDB: {e}')


def ensure_admin_user():
    """Create an admin user in MongoDB users collection if not present, using ADMIN_* env vars."""
    try:
        admin_username = os.getenv('ADMIN_USERNAME')
        admin_password = os.getenv('ADMIN_PASSWORD')
        admin_role = os.getenv('ADMIN_ROLE', 'CEO')
        if not (mongo_db and admin_username and admin_password):
            return
        users_col = mongo_db.get_collection('users')
        existing = users_col.find_one({'username': admin_username})
        if existing:
            print('Admin user already exists in MongoDB')
            return
        # store hashed password
        hashed = generate_password_hash(admin_password)
        users_col.insert_one({'username': admin_username, 'password': hashed, 'role': admin_role, 'created_at': datetime.utcnow()})
        print('Admin user created in MongoDB')
    except Exception as e:
        print(f'Failed to ensure admin user: {e}')

# Attempt to create admin user on startup
ensure_admin_user()

# Ensure upload and output folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Debug template loader
print("Template folder:", app.template_folder)
print("Available templates:", app.jinja_env.list_templates())

# ✅ LangChain DataFrame Agent Setup
# Dictionary to store agents for each user/session
dataframe_agents = {}

# ✅ Conversation Memory for Customer Care Agent
# Stores conversation history per user session
conversation_memories = {}  # Key: user_id, Value: ConversationBufferMemory

def get_or_create_memory(user_id):
    """Get or create a SimpleMemory for a user."""
    if user_id not in conversation_memories:
        conversation_memories[user_id] = SimpleMemory(
            memory_key="chat_history"
        )
    return conversation_memories[user_id]

def create_agent_for_dataframe(df, file_id):
    """Store dataframe for later queries (simple approach without experimental agent)."""
    try:
        # Store the dataframe for query processing
        dataframe_agents[file_id] = {"dataframe": df, "agent_type": "simple"}
        return {"dataframe": df, "agent_type": "simple"}
    except Exception as e:
        print(f"Error creating dataframe agent: {e}")
        return None

def query_dataframe_agent(query, file_id):
    """Execute a natural language query on the uploaded dataframe using simple pandas operations."""
    try:
        if file_id not in dataframe_agents:
            if file_id in uploaded_data:
                df = uploaded_data[file_id]
                create_agent_for_dataframe(df, file_id)
            else:
                return {"error": "No data available. Please upload a CSV file first."}
        
        agent_info = dataframe_agents[file_id]
        df = agent_info["dataframe"]
        
        # Simple query processing - use LLM to generate analysis
        response = llm.invoke(f"Analyze this dataframe based on the query. Query: {query}\n\nDataframe columns: {df.columns.tolist()}\n\nDataframe info:\n{df.describe().to_string()}")
        output = response.content if hasattr(response, 'content') else str(response)
        
        return {"result": output}
    except Exception as e:
        return {"error": f"Error processing query: {str(e)}"}


def detect_visualization_type(query):
    """Detect if a query requires visualization and return the chart type."""
    query_lower = query.lower()
    
    # Check if visualization is requested
    has_viz_request = any(kw in query_lower for kw in VISUALIZATION_KEYWORDS)
    if not has_viz_request:
        return None
    
    # Detect chart type
    for chart_type, keywords in VISUALIZATION_TYPES.items():
        if any(kw in query_lower for kw in keywords):
            return chart_type
    
    # Default to bar chart if visualization requested but type not specific
    return 'bar'


def generate_chart_from_dataframe(df, query, chart_type='bar'):
    """
    Generate a Plotly chart based on the dataframe and query.
    
    Returns: Plotly chart JSON or None if chart cannot be generated
    """
    try:
        query_lower = query.lower()
        
        # Detect what columns to use
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not numeric_cols:
            return None
        
        # Try to detect the relevant column from query
        primary_col = numeric_cols[0]
        group_col = None
        
        # Check if query mentions specific columns
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in query_lower:
                if col in numeric_cols:
                    primary_col = col
                elif col in categorical_cols:
                    group_col = col
        
        # If no group column detected, try to use categorical columns
        if not group_col and categorical_cols:
            group_col = categorical_cols[0]
        
        # Generate appropriate chart
        if chart_type == 'pie' and group_col:
            grouped = df.groupby(group_col)[primary_col].sum().reset_index()
            fig = px.pie(grouped, values=primary_col, names=group_col, 
                        title=f'{primary_col} Distribution by {group_col}')
        
        elif chart_type == 'bar' and group_col:
            grouped = df.groupby(group_col)[primary_col].agg(['mean', 'count']).reset_index()
            fig = px.bar(grouped, x=group_col, y='mean',
                        title=f'Average {primary_col} by {group_col}',
                        labels={'mean': f'Average {primary_col}'})
        
        elif chart_type == 'line' and group_col:
            grouped = df.groupby(group_col)[primary_col].mean().reset_index()
            fig = px.line(grouped, x=group_col, y=primary_col,
                         markers=True,
                         title=f'{primary_col} Trend')
        
        elif chart_type == 'scatter' and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                            title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
        
        elif chart_type == 'histogram':
            fig = px.histogram(df, x=primary_col,
                             title=f'Distribution of {primary_col}',
                             nbins=20)
        
        else:
            # Fallback to bar chart with first numeric column
            if len(numeric_cols) >= 1:
                value_counts = df[numeric_cols[0]].value_counts().head(10).reset_index()
                fig = px.bar(value_counts, x=numeric_cols[0], y='count',
                            title=f'Top 10 {numeric_cols[0]} Values')
            else:
                return None
        
        # Convert to JSON
        chart_json = fig.to_json()
        return chart_json
    
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return None


def detect_region(query):
    """
    Detect if the query mentions a specific region.
    
    Returns: Region name or None
    """
    query_lower = query.lower()
    
    for region, keywords in COMMON_REGIONS.items():
        for keyword in keywords:
            if keyword in query_lower:
                return region
    
    return None


def filter_dataframe_by_region(df, region):
    """
    Filter dataframe by region if Region column exists.
    
    Returns: Filtered dataframe
    """
    # Find region column (case-insensitive)
    region_col = None
    for col in df.columns:
        if col.lower() in ['region', 'location', 'city', 'area']:
            region_col = col
            break
    
    if not region_col:
        return df
    
    # Filter by region
    filtered_df = df[df[region_col].str.lower().str.contains(region, case=False, na=False)]
    return filtered_df if len(filtered_df) > 0 else df


def translate_to_nontechnical(text):
    """
    Convert technical terms to non-technical language.
    
    Returns: Simplified text
    """
    result = text
    for technical, simple in NON_TECHNICAL_TERMS.items():
        # Replace both exact matches and variations
        result = result.replace(technical, simple)
        result = result.replace(technical.capitalize(), simple.capitalize())
        result = result.replace(technical.upper(), simple.upper())
    
    return result


def format_business_summary(df, region=None, query=None):
    """
    Generate a business-friendly summary of the data.
    
    Returns: Summary text
    """
    try:
        # Filter by region if specified
        if region:
            df = filter_dataframe_by_region(df, region)
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        summary_parts = []
        
        # Add region context
        if region:
            summary_parts.append(f"📍 **{region.title()} Market Insights:**\n")
        
        # Add customer count
        if 'CustomerID' in df.columns:
            customer_count = df['CustomerID'].nunique()
            summary_parts.append(f"👥 **{customer_count:,} customers** in this area")
        
        # Add key metrics in simple language
        if 'SalesAmount' in df.columns:
            total_sales = df['SalesAmount'].sum()
            avg_sales = df['SalesAmount'].mean()
            summary_parts.append(f"\n💰 **Sales Performance:**")
            summary_parts.append(f"   • Total revenue: KES {total_sales:,.0f}")
            summary_parts.append(f"   • Average per transaction: KES {avg_sales:,.0f}")
        
        if 'Spending_Score' in df.columns:
            avg_spending = df['Spending_Score'].mean()
            summary_parts.append(f"\n🛍️ **Customer Spending:** {avg_spending:.0f}/100 (higher = more active)")
        
        # Add performance status
        if 'Churn_Risk' in df.columns:
            high_risk = (df['Churn_Risk'] == 'High').sum()
            total = len(df)
            risk_pct = (high_risk / total) * 100 if total > 0 else 0
            summary_parts.append(f"\n⚠️ **At-Risk Customers:** {risk_pct:.1f}% need attention")
        
        summary_text = "\n".join(summary_parts)
        return translate_to_nontechnical(summary_text)
    
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "Summary generation is not available."

def suggest_strategy_dynamically(df, query, ai_response):
    """
    Suggest actionable strategies based on data insights and detected trends.
    Integrates dynamically into chat responses for customer care.
    Acts as a living guide for explaining specialized pages (CLV, RFM, etc.)
    """
    try:
        strategies = []
        query_lower = query.lower()
        
        # STRATEGY 1: At-Risk Customer Detection & Retention
        if any(keyword in query_lower for keyword in ['at-risk', 'at risk', 'churn', 'leaving', 'retention']):
            if 'Churn_Risk' in df.columns:
                high_risk = df[df['Churn_Risk'] == 'High']
                high_risk_count = len(high_risk)
                if high_risk_count > 0:
                    # Calculate percentage
                    risk_pct = (high_risk_count / len(df)) * 100
                    avg_value = high_risk['Annual_Income'].mean() if 'Annual_Income' in df.columns else 0
                    
                    strategies.append(
                        f"🚨 **At-Risk Customer Alert:**\n"
                        f"• {high_risk_count} customers ({risk_pct:.1f}%) are at high churn risk\n"
                        f"• Average value of at-risk segment: ${avg_value:,.0f}\n"
                        f"\n💡 **Recommended Actions:**\n"
                        f"  1. Launch immediate outreach: personalized emails, loyalty offers, dedicated support\n"
                        f"  2. Offer special discounts or exclusive perks to re-engage them\n"
                        f"  3. Schedule check-in calls with your top at-risk customers\n"
                        f"  4. Create a win-back campaign with 30-day limited offers\n"
                        f"  5. Review why they're at risk: product issues? price concerns? poor service?"
                    )
            elif 'Spending_Score' in df.columns:
                low_spenders = df[df['Spending_Score'] < 30]
                if len(low_spenders) > 0:
                    strategies.append(
                        f"⚠️ **Low Engagement Alert:**\n"
                        f"• {len(low_spenders)} customers show low engagement patterns\n"
                        f"\n💡 **Re-engagement Strategy:**\n"
                        f"  1. Segment by reason: haven't purchased in 60+ days? Low purchase frequency?\n"
                        f"  2. Send re-engagement campaign: 'We miss you' offers, exclusive previews\n"
                        f"  3. Offer free shipping, bonus loyalty points, or early access to sales\n"
                        f"  4. Conduct quick survey: Why did they stop buying?"
                    )
        
        # STRATEGY 2: CLV (Customer Lifetime Value) Explanation & Optimization
        if any(keyword in query_lower for keyword in ['clv', 'lifetime value', 'customer value', 'high value']):
            if 'SalesAmount' in df.columns and 'CustomerID' in df.columns:
                total_clv = df['SalesAmount'].sum()
                avg_clv = df['SalesAmount'].mean()
                high_value = df[df['SalesAmount'] > df['SalesAmount'].quantile(0.75)]
                
                strategies.append(
                    f"💰 **Customer Lifetime Value (CLV) Explained:**\n"
                    f"• **What it means:** Total revenue each customer generates with you\n"
                    f"• Your total CLV: ${total_clv:,.0f}\n"
                    f"• Average CLV per customer: ${avg_clv:,.0f}\n"
                    f"• Top 25% of customers contribute: ${high_value['SalesAmount'].sum():,.0f}\n"
                    f"\n📊 **What This Tells You:**\n"
                    f"  • {len(high_value)} customers ({(len(high_value)/len(df)*100):.1f}%) drive ~{(high_value['SalesAmount'].sum()/total_clv*100):.0f}% of revenue\n"
                    f"  • Focus retention efforts on top CLV customers—losing one costs you significantly\n"
                    f"\n🎯 **Next Steps:**\n"
                    f"  1. Visit CLV page to see detailed breakdown by customer segment\n"
                    f"  2. Identify your 'Champions' (highest CLV) and VIPs\n"
                    f"  3. Create premium experiences for top customers\n"
                    f"  4. Use CLV to guide marketing budget allocation"
                )
        
        # STRATEGY 3: RFM Segment Strategy
        if any(keyword in query_lower for keyword in ['champion', 'loyal', 'rfm', 'segment', 'frequency']):
            if 'Spending_Score' in df.columns:
                champions = df[df['Spending_Score'] >= 75]
                loyal = df[(df['Spending_Score'] >= 50) & (df['Spending_Score'] < 75)]
                at_risk = df[df['Spending_Score'] < 30]
                
                strategies.append(
                    f"👑 **Customer Segment Strategy:**\n"
                    f"\n**Champions ({len(champions)} customers - {(len(champions)/len(df)*100):.0f}%):**\n"
                    f"  • Your best customers—they buy frequently and spend generously\n"
                    f"  • Strategy: VIP treatment, exclusive access, loyalty rewards\n"
                    f"  • Action: Personalized outreach, early sale access, premium support\n"
                    f"\n**Loyal Customers ({len(loyal)} customers - {(len(loyal)/len(df)*100):.0f}%):**\n"
                    f"  • Regular buyers with consistent engagement\n"
                    f"  • Strategy: Keep them engaged and moving toward Champion status\n"
                    f"  • Action: Loyalty programs, cross-sell opportunities, special offers\n"
                    f"\n**At-Risk ({len(at_risk)} customers - {(len(at_risk)/len(df)*100):.0f}%):**\n"
                    f"  • Low engagement—they may be close to leaving\n"
                    f"  • Strategy: Win-back campaigns with special incentives\n"
                    f"  • Action: Targeted discounts, free consultations, personalized recommendations"
                )
        
        # STRATEGY 4: Revenue Optimization
        if any(keyword in query_lower for keyword in ['revenue', 'sales', 'growth', 'increase', 'optimize']):
            if 'SalesAmount' in df.columns:
                total_revenue = df['SalesAmount'].sum()
                avg_transaction = df['SalesAmount'].mean()
                customer_count = df['CustomerID'].nunique() if 'CustomerID' in df.columns else len(df)
                
                strategies.append(
                    f"💹 **Revenue Optimization Strategy:**\n"
                    f"• Current total revenue: ${total_revenue:,.0f}\n"
                    f"• Average transaction value: ${avg_transaction:,.0f}\n"
                    f"• Customer count: {customer_count}\n"
                    f"\n**To grow revenue, focus on 3 areas:**\n"
                    f"  1. **Increase transaction value** (${avg_transaction:,.0f} → ${avg_transaction*1.2:,.0f}):\n"
                    f"     - Bundle products, upsell premium options\n"
                    f"     - Create 'product bundles' that increase order value\n"
                    f"\n  2. **Increase purchase frequency**:\n"
                    f"     - Email campaigns every 2-4 weeks\n"
                    f"     - Loyalty rewards for repeat purchases\n"
                    f"     - Subscription/subscription-like models\n"
                    f"\n  3. **Acquire new customers**:\n"
                    f"     - Marketing budget allocation based on high-CLV segment profile\n"
                    f"     - Target lookalike audiences of your Champions"
                )
        
        # STRATEGY 5: Default comprehensive overview
        if not strategies:
            if 'CustomerID' in df.columns and len(df) > 0:
                total_customers = df['CustomerID'].nunique()
                total_revenue = df['SalesAmount'].sum() if 'SalesAmount' in df.columns else 0
                
                strategies.append(
                    f"📈 **Dashboard Overview & Recommendations:**\n"
                    f"\n**Your Customer Base:**\n"
                    f"• Total customers: {total_customers}\n"
                    f"• Total transactions: {len(df)}\n"
                    f"• Total revenue: ${total_revenue:,.0f}\n"
                    f"\n**Explore Your Data:**\n"
                    f"  • Go to **RFM Analysis** to see customer segments (Champions, Loyal, At-Risk)\n"
                    f"  • Go to **CLV Analysis** to understand customer lifetime value\n"
                    f"  • Go to **Behavior** to see purchase patterns\n"
                    f"  • Go to **Churn Analysis** to identify customers likely to leave\n"
                    f"\n**Start with these questions:**\n"
                    f"  1. 'Who are my at-risk customers?'\n"
                    f"  2. 'What's my top customer segment?'\n"
                    f"  3. 'Explain the CLV results in simple terms'\n"
                    f"  4. 'What's my revenue breakdown by customer segment?'"
                )
        
        if strategies:
            return "\n\n".join(strategies)
        return ""
    
    except Exception as e:
        print(f"Error suggesting strategy: {str(e)}")
        return ""

# Simulated upload progress tracking
upload_progress = {}

def track_upload(file_id):
    for percent in range(0, 101, 10):
        if file_id in upload_progress:
            upload_progress[file_id] = percent
            time.sleep(0.5)  # Simulate upload time
    if file_id in upload_progress:
        del upload_progress[file_id]

def generate_progress(file_id):
    while file_id in upload_progress:
        percent = upload_progress.get(file_id, 0)
        print(f"Sending progress: {percent}% for file_id: {file_id}")
        yield f"data: {percent}\n\n"
        time.sleep(0.5)
    print(f"Completed upload for file_id: {file_id}")
    yield "data: 100\n\n"
    yield "event: complete\ndata: Upload complete!\n\n"


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple login route using hashed passwords stored in users.json.

    - Loads users from `app.config['USERS_FILE']` (JSON) on demand.
    - Expects JSON structure: {"username": {"password": "<hashed>", "role": "CEO"}}.
    - On successful login stores `session['user']` and `session['role']`.
    """
    # Load users from file if not in memory
    def load_users():
        users = app.config.get('USERS')
        if users is None:
            users = {}
            try:
                if os.path.exists(app.config['USERS_FILE']):
                    with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                        users = json.load(f)
            except Exception:
                users = {}
            app.config['USERS'] = users
        return app.config['USERS']

    def save_users(users):
        try:
            with open(app.config['USERS_FILE'], 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2)
        except Exception:
            pass

    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')
        if not username or not password:
            error = 'Please provide username and password.'
            return render_template('login.html', error=error)

        users = load_users()
        user_entry = users.get(username)
        if user_entry and 'password' in user_entry:
            # stored as hashed password
            if check_password_hash(user_entry['password'], password):
                if role not in ['CEO', 'Company Analyst']:
                    error = 'Invalid role selected.'
                    return render_template('login.html', error=error)
                    
                # Set session variables
                session['user'] = username
                session['role'] = role
                session['login_time'] = datetime.now().isoformat()
                
                # Redirect to index after successful login (regardless of role)
                return redirect(url_for('home'))
            else:
                error = 'Invalid username or password.'
                return render_template('login.html', error=error)

        # If no users configured or username not found, deny login (require registration)
        error = 'Invalid username or password. Please register first.'
        return render_template('login.html', error=error)

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    # Clear all session data
    session.clear()
    return redirect(url_for('login'))


@app.route('/ask_data', methods=['POST'])
@login_required
def ask_data():
    """
    Natural language query interface for the uploaded dataframe with security guardrails.
    
    ✅ Features:
    - Read-only access (no data modifications)
    - Restricted to analytics queries only
    - Input validation and filtering
    - Safe dataframe operations
    
    Example queries:
    - "What is the average spending score?"
    - "Show me the top 5 customers by annual income"
    - "How many unique regions are there?"
    - "Filter customers with spending score above 70"
    """
    try:
        # Get the query from the request
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Please provide a query."})
        
        # ✅ Security Check: Filter dangerous operations using global config
        query_lower = query.lower()
        for keyword in DANGEROUS_KEYWORDS:
            if keyword in query_lower:
                return jsonify({
                    "error": "❌ Security Policy Violation: Data modification operations are not allowed. I have read-only access to your data. Please ask analytical questions instead."
                })
        
        # Get the file_id from session
        file_id = session.get('last_uploaded_file_id')
        if not file_id:
            return jsonify({"error": "No data uploaded. Please upload a CSV file first."})
        
        # ✅ Enhanced safety wrapper for dataframe agent
        enhanced_query = f"""
IMPORTANT: You have READ-ONLY access to this dataframe. You cannot modify, delete, or change any data.

User Query: {query}

Constraints:
1. Only perform read-only operations (describe, head, tail, groupby, agg, filter, sort, count, mean, etc.)
2. Do NOT attempt to modify the dataframe (no assignments, no deletions, no updates)
3. If the user asks to modify data, refuse and explain you have read-only access
4. Provide analysis, insights, and summaries instead
5. Return results in a clear, human-readable format

Proceed with analysis:
"""
        
        # Query the agent with enhanced safety
        response = query_dataframe_agent(enhanced_query, file_id)
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})



@app.route('/ask_data_ui')
@login_required
def ask_data_ui():
    """Render the Ask Data UI page."""
    if 'last_uploaded_file_id' not in session:
        return render_template('ask_data.html', error="No data uploaded. Please upload a CSV file first.")
    
    return render_template('ask_data.html')


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """
    Chat endpoint for floating chat bubble with security guardrails.
    
    Features:
    - Read-only access (no data modifications allowed)
    - Restricted to customer analytics topics only
    - Prompt engineering to refuse harmful requests
    - Input validation and filtering
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        user_id = session.get('user')  # Get current user ID from session
        
        if not user_message:
            return jsonify({"error": "Please provide a message."})
        
        # ✅ Get or create conversation memory for this user (Customer Care)
        memory = get_or_create_memory(user_id)
        
        # ✅ Security Check 1: Filter malicious keywords using global config
        user_message_lower = user_message.lower()
        for keyword in DANGEROUS_KEYWORDS:
            if keyword in user_message_lower:
                return jsonify({
                    "response": "❌ I cannot perform data modification operations. I have read-only access to your data. I can only help you analyze and understand your customer data.",
                    "has_data": False,
                    "security_filtered": True
                })
        
        # ✅ Security Check 2: Check if query is analytics-related
        # Check if message contains at least one analytics keyword
        has_analytics_keyword = any(keyword in user_message_lower for keyword in ANALYTICS_KEYWORDS)
        
        if not has_analytics_keyword:
            # Check if it's a general greeting or platform question
            has_greeting = any(keyword in user_message_lower for keyword in GREETING_KEYWORDS)
            
            if not has_greeting:
                return jsonify({
                    "response": "ℹ️ I'm specialized in customer data analysis. I can help with questions about:\n• Customer metrics and statistics\n• Data patterns and trends\n• Segment analysis\n• Revenue and spending insights\n\nPlease ask me about your customer data!",
                    "has_data": False,
                    "out_of_scope": True
                })
        
        # Check if user has uploaded data
        file_id = session.get('last_uploaded_file_id')
        has_data = file_id and file_id in uploaded_data
        
        # ✅ Detect if query is region-specific
        detected_region = None
        if has_data:
            detected_region = detect_region(user_message)
        
        # Build context for the LLM with security instructions
        context = ""
        if has_data:
            df = uploaded_data[file_id]
            
            # Filter by region if detected
            if detected_region:
                df_context = filter_dataframe_by_region(df, detected_region)
            else:
                df_context = df
            
            # Provide data context to the LLM
            context = f"\n\nUser has uploaded a dataset with {len(df_context)} rows and columns: {', '.join(df_context.columns.tolist())}"
            context += f"\nData summary:\n{df_context.describe().to_string()}"
            
            # Add business summary for non-technical understanding
            business_summary = format_business_summary(df_context, detected_region, user_message)
            context += f"\n\nBusiness Summary:\n{business_summary}"
        
        # ✅ Smart Segmentation Support Assistant - Customer Care Persona
        system_prompt = """You are the Smart Segmentation Support Assistant, a polite and professional customer care representative. Your goal is to help business owners understand their customer data.

TONE & PERSONALITY:
• Always be courteous, patient, and helpful
• Use phrases like 'I would be happy to help,' 'Certainly,' and 'Please let me know if you need more details'
• Adopt a warm, consultative approach

CONTEXT & EXPERTISE:
• You have access to customer dataset and analytics
• Provide factual answers grounded in the data
• Explain insights in simple, business-friendly terms
• Avoid technical jargon—use everyday language

CONSULTANT APPROACH:
• When data shows negative trends (like high churn), deliver the news gently
• Immediately follow up with helpful strategies and recommendations
• Focus on actionable insights and solutions
• Build confidence and support informed decision-making

SAFETY & BOUNDARIES:
• Never perform 'Write', 'Delete', or 'Update' operations
• If a user asks to modify data, politely explain that you are a read-only assistant
• You are designed for analysis and support only
• Redirect requests for data modification with a helpful alternative

COMMUNICATION STYLE:
• Use simple, business-friendly language
• Break complex insights into clear bullet points
• Use emojis for clarity (👥 customers, 💰 revenue, ⚠️ risks, 🎯 opportunities)
• Provide context: explain what the numbers mean and why they matter
• Suggest next steps when appropriate"""
        
        full_message = f"{system_prompt}{context}\n\nUser message: {user_message}"
        
        # ✅ Store user message in conversation memory
        memory.save_context({"input": user_message}, {"output": ""})
        
        # ✅ Build enhanced context with conversation history for follow-up questions
        chat_history = memory.load_memory_variables({}).get("chat_history", "")
        if chat_history:
            full_message = f"{system_prompt}{context}\n\nConversation history:\n{chat_history}\n\nLatest user message: {user_message}"
        
        # Use the LLM to generate response
        response = llm.invoke(full_message)
        
        # Extract text from response
        ai_response = response.content if hasattr(response, 'content') else str(response)
        
        # ✅ Save AI response to memory for future context
        memory.save_context({"input": user_message}, {"output": ai_response})
        
        # ✅ Dynamic Strategy Integration - Suggest actionable strategies based on data
        suggested_strategy = ""
        if has_data:
            try:
                df = uploaded_data[file_id]
                suggested_strategy = suggest_strategy_dynamically(df, user_message, ai_response)
                if suggested_strategy:
                    ai_response = f"{ai_response}\n\n{suggested_strategy}"
            except Exception as e:
                print(f"⚠️ Could not generate strategy suggestion: {str(e)}")
        
        # ✅ Visualization Generation: Check if user wants a chart
        chart_json = None
        chart_type = None
        
        if has_data:
            chart_type = detect_visualization_type(user_message)
            if chart_type:
                try:
                    df = uploaded_data[file_id]
                    # Apply regional filtering if applicable
                    region = detect_region(user_message)
                    if region:
                        df = filter_dataframe_by_region(df, region)
                        if df.empty:
                            print(f"⚠️ No data found for region: {region}")
                    chart_json = generate_chart_from_dataframe(df, user_message, chart_type)
                    if chart_json:
                        print(f"✅ Generated {chart_type} chart for query: {user_message}")
                except Exception as e:
                    print(f"⚠️ Could not generate chart: {str(e)}")
        
        response_data = {
            "response": ai_response,
            "has_data": has_data
        }
        
        # Include chart if generated
        if chart_json:
            response_data["chart"] = chart_json
            response_data["chart_type"] = chart_type
        
        return jsonify(response_data)
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": f"Error processing message: {str(e)}"})




@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register new users and save them to a users.json file.

    The stored format is:
      {"username": {"password": "<hashed>", "role": "CEO"}}
    """

    def load_users():
        users = app.config.get('USERS')
        if users is None:
            users = {}
            try:
                if os.path.exists(app.config['USERS_FILE']):
                    with open(app.config['USERS_FILE'], 'r', encoding='utf-8') as f:
                        users = json.load(f)
            except Exception:
                users = {}
            app.config['USERS'] = users
        return app.config['USERS']

    def save_users(users):
        try:
            with open(app.config['USERS_FILE'], 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2)
        except Exception:
            pass

    error = None

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role')

        if not username or not password or not role:
            error = 'Please fill all fields.'
            return render_template('register.html', error=error)

        if role not in ('CEO', 'Company Analyst'):
            error = 'Invalid role selected.'
            return render_template('register.html', error=error)

        users = load_users()
        if username in users:
            error = 'Username already exists. Choose another.'
            return render_template('register.html', error=error)

        # Save new user
        hashed = generate_password_hash(password)
        users[username] = {'password': hashed, 'role': role}
        save_users(users)

        # Set session and redirect to index
        session['user'] = username
        session['role'] = role
        session['login_time'] = datetime.now().isoformat()

        return redirect(url_for('home'))

    # GET request: just render registration form
    return render_template('register.html', error=None)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

# Consolidated home route
@app.route('/', methods=['GET', 'POST'])
def home():
    # Redirect to login if not authenticated
    if 'user' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        if 'file' not in request.files:
            print("No file part in request")
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            file_id = id(file)
            upload_progress[file_id] = 0
            threading.Thread(target=track_upload, args=(file_id,), daemon=True).start()
            print(f"Processing file: {file.filename} with file_id: {file_id}")
            try:
                df = pd.read_csv(file)
                uploaded_data[file_id] = df
                session['last_uploaded_file_id'] = file_id
                
                # ✅ Create a dataframe agent for this file
                agent = create_agent_for_dataframe(df, file_id)
                if agent:
                    print(f"✅ Dataframe agent created for file_id: {file_id}")
                else:
                    print(f"⚠️ Warning: Could not create dataframe agent")
                
                print("File processed and stored successfully")
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                return render_template('index.html', error=f"Error processing file: {str(e)}")
            return Response(
                generate_progress(file_id),
                mimetype='text/event-stream'
            )
    return render_template('index.html')

@app.route('/overview')
def overview_analysis():
    error = None
    raw_data = None
    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]
        print(f"Columns available in overview: {df.columns.tolist()}")
        raw_data = df  # Assign full DataFrame without truncation
    else:
        error = "No data available. Please upload a CSV file first."
        print(f"Error in overview: {error}")
    return render_template('overview.html', error=error, raw_data=raw_data)

@app.route('/demographic_trends', endpoint='demographic_trends_analysis')
@login_required
def demographic_trends():
    error = None
    visualizations = []
    demographic_trends_results = None
    output_file = None
    filename = None
    download_link = None
    selected_month = request.args.get('month', 'All')
    month_list = []

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]

        if 'OrderDate' not in df.columns:
            if 'PurchaseDate' in df.columns:
                df.rename(columns={'PurchaseDate': 'OrderDate'}, inplace=True)
            else:
                error = "CSV must contain 'OrderDate' or 'PurchaseDate' for time filtering."
                return render_template(
                    'demographic.html',
                    error=error,
                    visualizations=[],
                    demo_results=None,
                    selected_month=selected_month,
                    month_list=[],
                    download_link=None
                )

        df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
        df.dropna(subset=['OrderDate'], inplace=True)
        df['Month'] = df['OrderDate'].dt.to_period('M').astype(str)
        month_list = sorted(df['Month'].unique().tolist())

        if selected_month != 'All':
            df = df[df['Month'] == selected_month]

        # Import your trend function
        from demographic import calculate_demographic_trends
        trends, output_file, visualizations, error = calculate_demographic_trends(df)

        if trends and not error:
            # Create a single DataFrame to represent all trends
            numeric_trends = {k: pd.Series(v['trend']) for k, v in trends.items() if v['type'] == 'numeric'}
            combined_df = pd.DataFrame(numeric_trends).reset_index()
            demographic_trends_results = combined_df.to_html(
                classes='table table-bordered table-striped display',
                index=False,
                table_id='demographicTrendsTable'
            )
            filename = os.path.basename(output_file)
            download_link = url_for('download_demographic', filename=filename)
        else:
            demographic_trends_results = None
    else:
        error = "No uploaded CSV found. Please upload a file first."

    return render_template(
        'demographic.html',
        error=error,
        visualizations=visualizations,
        demo_results=demographic_trends_results,
        selected_month=selected_month,
        month_list=month_list,
        download_link=download_link
    )


@app.route('/download_demographic/<filename>')
def download_demographic(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return f"File {filename} not found.", 404


def load_uploaded_data():
    """Return the last uploaded DataFrame or an empty DataFrame.

    Uses the global `uploaded_data` dict and `session['last_uploaded_file_id']`.
    """
    try:
        file_id = session.get('last_uploaded_file_id')
        if file_id and file_id in uploaded_data:
            return uploaded_data[file_id]
    except Exception:
        pass
    return pd.DataFrame()


@app.route('/ceo/business_strategies')
@login_required
@role_required('CEO')
def ceo_business_strategies():
    data = load_uploaded_data()
    if data.empty:
        return render_template('error.html', error='No data', message='No uploaded dataset found. Please upload a CSV first.')

    # Guard: ensure required columns exist
    required = {'SalesAmount', 'CustomerID'}
    if not required.issubset(set(data.columns)):
        return render_template('error.html', error='Missing columns', message=f'Required columns missing: {required - set(data.columns)}')

    # Calculate high-level metrics
    revenue = data['SalesAmount'].sum()
    orders = len(data)
    customers = data['CustomerID'].nunique()

    insights = {
        'revenue': revenue,
        'orders': orders,
        'customers': customers
    }

    strategy = f"""
    Based on current business performance:
    - Total revenue: ${revenue:,.2f}
    - Total customers: {customers}
    - Total orders: {orders}

    ✅ AI-Driven Strategies:
    1.Launch a targeted loyalty and rewards program to strengthen repeat purchase behavior and boost customer lifetime value.
    2. Deploy personalized retention offers to proactively reduce churn in at-risk segments.
    3. Scale investment in your highest-performing product categories, leveraging momentum to capture additional market share.
    4. Prioritize expansion in high-growth regions where demand velocity is outpacing the average market trend.
    5. Apply dynamic pricing and discount optimization to improve turnover for slow-moving categories.
    """

    return render_template('ceo_business_strategies.html', insights=insights, strategy=strategy)


@app.route('/ceo/financial_trends')
@login_required
@role_required('CEO')
def ceo_financial_trends():
    data = load_uploaded_data()
    if data.empty:
        return render_template('error.html', error='No data', message='No uploaded dataset found. Please upload a CSV first.')

    if 'PurchaseDate' not in data.columns or 'SalesAmount' not in data.columns:
        return render_template('error.html', error='Missing columns', message="Required columns 'PurchaseDate' or 'SalesAmount' missing.")

    data = data.copy()
    data['PurchaseDate'] = pd.to_datetime(data['PurchaseDate'], errors='coerce')
    data.dropna(subset=['PurchaseDate'], inplace=True)

    monthly = data.groupby(data['PurchaseDate'].dt.to_period('M')).agg({
        'SalesAmount': 'sum'
    }).reset_index()
    monthly['PurchaseDate'] = monthly['PurchaseDate'].astype(str)

    fig = px.line(monthly, x='PurchaseDate', y='SalesAmount', title='Monthly Revenue Trend')
    chart_html = fig.to_html(full_html=False)

    return render_template('ceo_financial_trends.html', chart_html=chart_html)


@app.route('/ceo/performance_metrics')
@login_required
@role_required('CEO')
def ceo_performance_metrics():
    data = load_uploaded_data()
    if data.empty:
        return render_template('error.html', error='No data', message='No uploaded dataset found. Please upload a CSV first.')

    required = {'SalesAmount', 'CustomerID', 'ProductCategory'}
    if not required.issubset(set(data.columns)):
        # allow partial metrics but notify
        missing = required - set(data.columns)
        return render_template('error.html', error='Missing columns', message=f'Required columns missing: {missing}')

    metrics = {
        'avg_order_value': float(data['SalesAmount'].mean()),
        'purchase_frequency': float(len(data) / data['CustomerID'].nunique()) if data['CustomerID'].nunique() > 0 else 0,
        'customer_retention': float(data['CustomerID'].value_counts().mean()),
        'top_product': data['ProductCategory'].value_counts().idxmax() if 'ProductCategory' in data.columns and not data['ProductCategory'].empty else None
    }

    return render_template('ceo_performance_metrics.html', metrics=metrics)


@app.route('/dashboard')
def dashboard():
    # Require authentication
    if 'user' not in session:
        return redirect(url_for('login'))

    role = session.get('role')
    view = request.args.get('view', 'analyst' if role == 'Company Analyst' else 'executive')
    
    # Prepare dashboard data based on role
    context = {
        'user': session.get('user'),
        'role': role,
        'login_time': session.get('login_time'),
        'view': view
    }
    
    if role == 'CEO':
        # Add executive metrics/summaries
        # TODO: Add business metrics, financial trends, etc.
        pass
    elif role == 'Company Analyst':
        # Add analyst specific data
        # TODO: Add detailed analytics metrics
        pass
        
    return render_template('dashboard.html', **context)



@app.route('/rfm', methods=['GET', 'POST'])
@login_required
def rfm_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    table = None
    download_link = None
    selected_segment = request.values.get('segment', 'All')
    analysis_date_str = request.form.get('analysis_date') or request.args.get('analysis_date')
    analysis_date = pd.to_datetime(analysis_date_str) if analysis_date_str else None

    if 'last_uploaded_file_id' not in session or session['last_uploaded_file_id'] not in uploaded_data:
        error = "No data available. Please upload a CSV file first."
        return render_template('rfm.html', error=error, selected_segment=selected_segment, analysis_date=analysis_date_str)

    df = uploaded_data[session['last_uploaded_file_id']]
    print(f"Columns available in rfm: {df.columns.tolist()}")

    if request.method == 'POST' and analysis_date:
        required_columns = ['CustomerID', 'OrderDate', 'Annual_Income']
        if not all(col in df.columns for col in required_columns):
            error = "Uploaded CSV must contain CustomerID, OrderDate, and Annual_Income columns."
            return render_template('rfm.html', error=error, selected_segment=selected_segment, analysis_date=analysis_date_str)

        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        current_date = analysis_date

        rfm_table = df.groupby(['CustomerID']).agg({
            'OrderDate': lambda x: (current_date - x.max()).days,
            'CustomerID': 'count',
            'Annual_Income': 'sum'
        }).rename(columns={
            'OrderDate': 'Recency',
            'CustomerID': 'Frequency',
            'Annual_Income': 'Monetary'
        })

        try:
            r_bins = pd.qcut(rfm_table['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
            rfm_table['R_Score'] = r_bins
        except ValueError:
            rfm_table['R_Score'] = 2

        try:
            f_bins = pd.qcut(rfm_table['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
            rfm_table['F_Score'] = f_bins
        except ValueError:
            rfm_table['F_Score'] = 2

        try:
            m_bins = pd.qcut(rfm_table['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4], duplicates='drop')
            rfm_table['M_Score'] = m_bins
        except ValueError:
            rfm_table['M_Score'] = 2

        rfm_table['RFM_Score'] = rfm_table['R_Score'].astype(str) + rfm_table['F_Score'].astype(str) + rfm_table['M_Score'].astype(str)

        def segment_name(score):
            if score == '444':
                return 'Champions'
            elif score[0] == '4':
                return 'Loyal Customers'
            elif score[1] == '4':
                return 'Frequent Buyers'
            elif score[2] == '4':
                return 'Big Spenders'
            elif score[0] == '1':
                return 'At Risk'
            else:
                return 'Others'

        rfm_table['Segment'] = rfm_table['RFM_Score'].apply(segment_name)

        if 'FirstName' in df.columns and 'LastName' in df.columns:
            df['CustomerName'] = df['FirstName'] + ' ' + df['LastName']
        elif 'FirstName' in df.columns:
            df['CustomerName'] = df['FirstName']

        if 'CustomerName' in df.columns:
            customer_names = df[['CustomerID', 'CustomerName']].drop_duplicates().set_index('CustomerID')
            rfm_table = rfm_table.merge(customer_names, left_index=True, right_index=True, how='left')
        else:
            rfm_table['CustomerName'] = 'N/A'

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        clustering_data = rfm_table[['Recency', 'Frequency', 'Monetary']].dropna()
        if not clustering_data.empty:
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(clustering_data)
            kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
            rfm_table.loc[clustering_data.index, 'Cluster'] = kmeans.fit_predict(rfm_scaled)
        else:
            rfm_table['Cluster'] = None

        rfm_table = rfm_table[['CustomerName', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'Segment', 'Cluster']]

        if selected_segment != 'All':
            rfm_table = rfm_table[rfm_table['Segment'] == selected_segment]

        csv_path = os.path.join(app.static_folder, 'rfm_output.csv')
        rfm_table.to_csv(csv_path)
        download_link = url_for('static', filename='rfm_output.csv')

        table = rfm_table.to_html(
            classes='table table-bordered table-striped table-hover display',
            index=True,
            table_id="rfmTable"
        )
        table += '''
        <script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
        <script>
        $(document).ready(function() {
            $('#rfmTable').DataTable();
        });
        </script>
        '''

        segment_counts = rfm_table['Segment'].value_counts().sort_values(ascending=False)
        pie_chart_data = {
            'data': [{
                'values': segment_counts.values.tolist(),
                'labels': segment_counts.index.tolist(),
                'type': 'pie'
            }],
            'layout': {
                'title': 'RFM Segment Distribution'
            }
        }
        pie_chart = f'''
            <div id="pie-chart" style="width: 100%; height: 400px;"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>Plotly.newPlot("pie-chart", {json.dumps(pie_chart_data)});</script>
        '''

        avg_monetary = [
            rfm_table[rfm_table['Segment'] == seg]['Monetary'].mean()
            for seg in segment_counts.index
        ]
        bar_chart_data = {
            'data': [{
                'x': segment_counts.index.tolist(),
                'y': avg_monetary,
                'type': 'bar',
                'text': [f"${val:.2f}" for val in avg_monetary],
                'textposition': 'auto'
            }],
            'layout': {
                'title': 'Average Monetary by Segment',
                'yaxis': {'title': 'Average Monetary ($)'}
            }
        }
        bar_chart = f'''
            <div id="bar-chart" style="width: 100%; height: 400px;"></div>
            <script>Plotly.newPlot("bar-chart", {json.dumps(bar_chart_data)});</script>
        '''

    elif not analysis_date:
        error = "Please select an analysis date and submit the form."

    return render_template(
        'rfm.html',
        error=error,
        pie_chart=pie_chart,
        bar_chart=bar_chart,
        table=table,
        download_link=download_link,
        selected_segment=selected_segment,
        analysis_date=analysis_date_str
    )
@app.route('/behavior', methods=['GET', 'POST'])
def behavior_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    line_chart = None
    table = None
    download_link = None
    selected_segment = request.values.get('segment', 'All')

    if 'last_uploaded_file_id' not in session or session['last_uploaded_file_id'] not in uploaded_data:
        error = "No data available. Please upload a CSV file first."
        return render_template('behavior.html', error=error, selected_segment=selected_segment)

    df = uploaded_data[session['last_uploaded_file_id']]

    if not all(col in df.columns for col in ['CustomerID', 'OrderDate', 'OrderCount']):
        error = "CSV must contain 'CustomerID', 'OrderDate', and 'OrderCount' columns."
        return render_template('behavior.html', error=error, selected_segment=selected_segment)

    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['OrderMonth'] = df['OrderDate'].dt.to_period('M').astype(str)

    behavior_table = df.groupby('CustomerID').agg({
        'OrderCount': 'sum'
    }).rename(columns={'OrderCount': 'TotalOrders'}).reset_index()

    behavior_table['BehaviorType'] = behavior_table['TotalOrders'].apply(
        lambda x: 'Frequent Buyer' if x >= 5 else 'Two-Time Buyer' if x >= 2 else 'One-Time Buyer'
    )

    if 'FirstName' in df.columns and 'LastName' in df.columns:
        df['CustomerName'] = df['FirstName'] + ' ' + df['LastName']
    elif 'FirstName' in df.columns:
        df['CustomerName'] = df['FirstName']

    if 'CustomerName' in df.columns:
        names = df[['CustomerID', 'CustomerName']].drop_duplicates()
        behavior_table = behavior_table.merge(names, on='CustomerID', how='left')
    else:
        behavior_table['CustomerName'] = 'N/A'

    # KMeans clustering on behavior
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    clustering_features = behavior_table[['TotalOrders']].copy()
    clustering_features = clustering_features.fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clustering_features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    behavior_table['Cluster'] = kmeans.fit_predict(scaled)

    # Filter by segment
    if selected_segment != 'All':
        behavior_table = behavior_table[behavior_table['BehaviorType'] == selected_segment]

    # CSV Export
    csv_path = os.path.join(app.static_folder, 'behavior_output.csv')
    behavior_table.to_csv(csv_path, index=False)
    download_link = url_for('static', filename='behavior_output.csv')

    # Generate HTML table
    table = behavior_table.to_html(
        classes='table table-bordered table-striped table-hover display',
        index=False,
        table_id="behaviorTable"
    )
    table += '''
    <script src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>
    <script>
    $(document).ready(function() {
        $('#behaviorTable').DataTable({
            pageLength: 10,
            lengthChange: true,
            searching: true,
            ordering: true
        });
    });
    </script>
    '''

    # Pie Chart
    behavior_counts = behavior_table['BehaviorType'].value_counts()
    pie_data = {
        'data': [{
            'labels': behavior_counts.index.tolist(),
            'values': behavior_counts.values.tolist(),
            'type': 'pie'
        }],
        'layout': {'title': 'Customer Behavior Types'}
    }
    pie_chart = f'''
    <div id="behavior-pie" style="width:100%; height:400px;"></div>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>Plotly.newPlot("behavior-pie", {json.dumps(pie_data)});</script>
    '''

    # Bar Chart
    avg_order_counts = behavior_table.groupby('BehaviorType')['TotalOrders'].mean().sort_values(ascending=False)
    bar_data = {
        'data': [{
            'x': avg_order_counts.index.tolist(),
            'y': avg_order_counts.values.tolist(),
            'type': 'bar',
            'text': [f"{x:.2f}" for x in avg_order_counts.values],
            'textposition': 'auto'
        }],
        'layout': {
            'title': 'Avg Order Count per Behavior Type',
            'yaxis': {'title': 'Average Orders'}
        }
    }
    bar_chart = f'''
    <div id="behavior-bar" style="width:100%; height:400px;"></div>
    <script>Plotly.newPlot("behavior-bar", {json.dumps(bar_data)});</script>
    '''

    # Line Chart
    monthly_orders = df.groupby('OrderMonth')['CustomerID'].count()
    line_data = {
        'data': [{
            'x': monthly_orders.index.tolist(),
            'y': monthly_orders.values.tolist(),
            'mode': 'lines+markers',
            'type': 'scatter',
            'line': {'shape': 'linear'}
        }],
        'layout': {
            'title': 'Total Orders Over Time',
            'xaxis': {'title': 'Month'},
            'yaxis': {'title': 'Number of Orders'}
        }
    }
    line_chart = f'''
    <div id="behavior-line" style="width:100%; height:400px;"></div>
    <script>Plotly.newPlot("behavior-line", {json.dumps(line_data)});</script>
    '''

    segment_summary = {
        "Total Customers": behavior_table.shape[0],
        "Most Common Behavior Type": behavior_counts.idxmax(),
        "Most Common Count": behavior_counts.max(),
        "Unique Behavior Types": len(behavior_counts),
        "Average Orders per Customer": round(behavior_table['TotalOrders'].mean(), 2)
    }

    return render_template('behavior.html',
                           error=error,
                           pie_chart=pie_chart,
                           bar_chart=bar_chart,
                           line_chart=line_chart,
                           table=table,
                           download_link=download_link,
                           segment_summary=segment_summary,
                           selected_segment=selected_segment)

@app.route('/product_preference')
def product_preference_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    line_chart = None
    pref_results = None
    segment_summary = None

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]
        print(f"Columns available in product_preference: {df.columns.tolist()}")
        if 'Spending_Score' in df.columns and 'ProductCategory' in df.columns:
            category_counts = df['ProductCategory'].value_counts()
            pie_data = {
                'data': [{
                    'labels': category_counts.index.tolist(),
                    'values': category_counts.values.tolist(),
                    'type': 'pie'
                }],
                'layout': {'title': 'Product Category Distribution'}
            }
            pie_chart = f'''
                <div id="pie-chart" style="width:100%; height:400px;"></div>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>Plotly.newPlot("pie-chart", {json.dumps(pie_data)});</script>
            '''

            avg_scores = df.groupby('ProductCategory')['Spending_Score'].mean().sort_values(ascending=False)
            bar_data = {
                'data': [{
                    'x': avg_scores.index.tolist(),
                    'y': avg_scores.values.tolist(),
                    'type': 'bar',
                    'text': [f"{val:.2f}" for val in avg_scores.values],
                    'textposition': 'auto'
                }],
                'layout': {'title': 'Average Spending Score by Product Category'}
            }
            bar_chart = f'''
                <div id="bar-chart" style="width:100%; height:400px;"></div>
                <script>Plotly.newPlot("bar-chart", {json.dumps(bar_data)});</script>
            '''

            if 'OrderDate' in df.columns:
                df['OrderDate'] = pd.to_datetime(df['OrderDate'])
                df['OrderMonth'] = df['OrderDate'].dt.to_period('M').astype(str)
                monthly_orders = df.groupby('OrderMonth')['CustomerID'].count()
                line_data = {
                    'data': [{
                        'x': monthly_orders.index.tolist(),
                        'y': monthly_orders.values.tolist(),
                        'mode': 'lines+markers',
                        'type': 'scatter'
                    }],
                    'layout': {'title': 'Monthly Product Orders Over Time'}
                }
                line_chart = f'''
                    <div id="line-chart" style="width:100%; height:400px;"></div>
                    <script>Plotly.newPlot("line-chart", {json.dumps(line_data)});</script>
                '''

            pref_results = df.groupby('ProductCategory').agg({
                'CustomerID': 'count',
                'Spending_Score': 'mean'
            }).reset_index()
            pref_results.columns = ['Product Category', 'Customer Count', 'Average Spending Score']
            pref_results['Average Spending Score'] = pref_results['Average Spending Score'].round(2)

            segment_summary = {
                'Total Categories': len(category_counts),
                'Most Popular Category': category_counts.idxmax(),
                'Total Customers': df['CustomerID'].nunique(),
                'Avg Spending Score': round(df['Spending_Score'].mean(), 2)
            }
        else:
            error = "Uploaded CSV must contain Spending_Score and ProductCategory columns."
    else:
        error = "No data available. Please upload a valid CSV."

    return render_template(
        'product_preference.html',
        error=error,
        pie_chart=pie_chart,
        bar_chart=bar_chart,
        line_chart=line_chart,
        pref_results=pref_results,
        segment_summary=segment_summary
    )

@app.route('/clv')
@login_required
def clv_analysis():
    error = None
    pie_chart = None
    bar_chart = None
    line_chart = None
    scatter_plot = None
    clv_results = None
    clv_summary = None
    output_file = None
    download_link = None

    # ... rest of the function logic ...

    file_id = session.get('last_uploaded_file_id')
    if not file_id or file_id not in uploaded_data:
        error = "No data available. Please upload a CSV file first."
        return render_template(
            'clv.html',
            error=error,
            pie_chart=pie_chart,
            bar_chart=bar_chart,
            line_chart=line_chart,
            scatter_plot=scatter_plot,
            clv_results=clv_results,
            clv_summary=clv_summary,
            download_link=download_link
        )

    df = uploaded_data[file_id]
    if not all(col in df.columns for col in ['CustomerID', 'OrderDate', 'SalesAmount']):
        error = "CSV must contain 'CustomerID', 'OrderDate', and 'SalesAmount' columns."
        return render_template(
            'clv.html',
            error=error,
            pie_chart=pie_chart,
            bar_chart=bar_chart,
            line_chart=line_chart,
            scatter_plot=scatter_plot,
            clv_results=clv_results,
            clv_summary=clv_summary,
            download_link=download_link
        )

    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['OrderMonth'] = df['OrderDate'].dt.to_period('M').astype(str)

    clv_table = df.groupby('CustomerID').agg({
        'SalesAmount': 'sum',
        'OrderDate': 'count'
    }).rename(columns={'SalesAmount': 'CLV', 'OrderDate': 'OrderCount'}).reset_index()

    # Quartiles
    clv_table['CLV_Quartile'] = pd.qcut(
        clv_table['CLV'], 4,
        labels=['Q1 - Lowest', 'Q2 - Low-Mid', 'Q3 - High-Mid', 'Q4 - Highest']
    )

    # KMeans Clustering
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features = scaler.fit_transform(clv_table[['CLV', 'OrderCount']])
    kmeans = KMeans(n_clusters=4, random_state=0)
    clv_table['CLV_Cluster'] = kmeans.fit_predict(features)

    # Pie chart
    clv_brackets = pd.cut(clv_table['CLV'], bins=[0, 200, 500, 1000, 2000, float('inf')],
                          labels=['<200', '200-500', '500-1000', '1000-2000', '2000+'])
    pie_counts = clv_brackets.value_counts().sort_index()
    pie_data = {
        'data': [{
            'labels': pie_counts.index.tolist(),
            'values': pie_counts.values.tolist(),
            'type': 'pie'
        }],
        'layout': {'title': 'CLV Bracket Distribution'}
    }
    pie_chart = f'''
        <div id="clv-pie"></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>Plotly.newPlot("clv-pie", {json.dumps(pie_data)});</script>
    '''

    # Bar chart for top customers
    top_customers = clv_table.nlargest(10, 'CLV')
    bar_data = {
        'data': [{
            'x': top_customers['CustomerID'].astype(str).tolist(),
            'y': top_customers['CLV'].tolist(),
            'type': 'bar',
            'text': [f"${v:.2f}" for v in top_customers['CLV']],
            'textposition': 'auto'
        }],
        'layout': {'title': 'Top 10 Customers by CLV', 'yaxis': {'title': 'CLV ($)'}}
    }
    bar_chart = f'''
        <div id="clv-bar"></div>
        <script>Plotly.newPlot("clv-bar", {json.dumps(bar_data)});</script>
    '''

    # Line chart of monthly accumulation
    clv_monthly = df.groupby('OrderMonth')['SalesAmount'].sum().sort_index()
    line_data = {
        'data': [{
            'x': clv_monthly.index.tolist(),
            'y': clv_monthly.values.tolist(),
            'mode': 'lines+markers',
            'line': {'shape': 'linear'}
        }],
        'layout': {'title': 'CLV Accumulation Over Time', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Total Sales'}}
    }
    line_chart = f'''
        <div id="clv-line"></div>
        <script>Plotly.newPlot("clv-line", {json.dumps(line_data)});</script>
    '''

    # Scatter plot of CLV vs Order Count
    scatter_data = {
        'data': [{
            'x': clv_table['OrderCount'].tolist(),
            'y': clv_table['CLV'].tolist(),
            'mode': 'markers',
            'type': 'scatter',
            'marker': {
                'size': 10,
                'color': clv_table['CLV_Cluster'].tolist(),
                'colorscale': 'Viridis',
                'showscale': True
            }
        }],
        'layout': {'title': 'CLV vs Frequency', 'xaxis': {'title': 'Order Count'}, 'yaxis': {'title': 'CLV'}}
    }
    scatter_plot = f'''
        <div id="clv-scatter"></div>
        <script>Plotly.newPlot("clv-scatter", {json.dumps(scatter_data)});</script>
    '''

    # Summary
    clv_summary = {
        "Total Customers": len(clv_table),
        "Total CLV": f"${clv_table['CLV'].sum():,.2f}",
        "Average CLV": f"${clv_table['CLV'].mean():,.2f}",
        "Max CLV": f"${clv_table['CLV'].max():,.2f}",
        "Min CLV": f"${clv_table['CLV'].min():,.2f}"
    }

    # Apply segment filter if requested
    selected_segment = request.args.get('segment', 'All')
    if selected_segment != 'All':
        clv_table = clv_table[clv_table['CLV_Quartile'] == selected_segment]

    # Prepare download
    clv_results = clv_table.sort_values(by='CLV', ascending=False)
    output_file = f"clv_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
    clv_results.to_csv(output_path, index=False)
    download_link = url_for('download_results', filename=output_file)

    return render_template('clv.html',
                           error=error,
                           pie_chart=pie_chart,
                           bar_chart=bar_chart,
                           line_chart=line_chart,
                           scatter_plot=scatter_plot,
                           clv_results=clv_results,
                           clv_summary=clv_summary,
                           download_link=download_link)


@app.route('/churn')
@login_required
def churn_analysis():
    error = None
    churn_pie_chart = None
    churn_bar_chart = None
    churn_line_chart = None
    churn_summary = None
    churn_results = pd.DataFrame()  # Always defined to avoid template errors
    download_link = None

    file_id = session.get('last_uploaded_file_id')

    if file_id and file_id in uploaded_data:
        df = uploaded_data[file_id]

        if 'Spending_Score' in df.columns and 'CustomerID' in df.columns:
            df['Churn_Risk'] = pd.cut(
                df['Spending_Score'],
                bins=[0, 30, 70, 100],
                labels=['High', 'Medium', 'Low']
            )

            # Filter by selected risk level
            selected_risk = request.args.get('risk', 'All')
            if selected_risk != 'All':
                df = df[df['Churn_Risk'] == selected_risk]

            # Pie chart
            churn_counts = df['Churn_Risk'].value_counts().sort_index()
            pie_data = {
                'data': [{
                    'labels': churn_counts.index.tolist(),
                    'values': churn_counts.values.tolist(),
                    'type': 'pie'
                }],
                'layout': {'title': 'Churn Risk Distribution'}
            }
            churn_pie_chart = f'''
                <div id="churn-pie"></div>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>Plotly.newPlot("churn-pie", {json.dumps(pie_data)});</script>
            '''

            # Bar chart
            avg_spending = df.groupby('Churn_Risk')['Spending_Score'].mean()
            bar_data = {
                'data': [{
                    'x': avg_spending.index.tolist(),
                    'y': avg_spending.values.tolist(),
                    'type': 'bar',
                    'text': [f"{v:.1f}" for v in avg_spending],
                    'textposition': 'auto'
                }],
                'layout': {
                    'title': 'Avg Spending by Churn Risk',
                    'yaxis': {'title': 'Avg Spending Score'}
                }
            }
            churn_bar_chart = f'''
                <div id="churn-bar"></div>
                <script>Plotly.newPlot("churn-bar", {json.dumps(bar_data)});</script>
            '''

            # Line chart
            sorted_df = df.sort_values(by='CustomerID')
            line_data = {
                'data': [{
                    'x': sorted_df['CustomerID'].astype(str).tolist(),
                    'y': sorted_df['Spending_Score'].tolist(),
                    'mode': 'lines+markers'
                }],
                'layout': {
                    'title': 'Spending Score Trend by Customer',
                    'xaxis': {'title': 'CustomerID'},
                    'yaxis': {'title': 'Spending Score'}
                }
            }
            churn_line_chart = f'''
                <div id="churn-line"></div>
                <script>Plotly.newPlot("churn-line", {json.dumps(line_data)});</script>
            '''

            # Summary
            churn_summary = {
                'Total Customers': len(df),
                'High Risk': int((df['Churn_Risk'] == 'High').sum()),
                'Medium Risk': int((df['Churn_Risk'] == 'Medium').sum()),
                'Low Risk': int((df['Churn_Risk'] == 'Low').sum())
            }

            # Results Table (with optional CustomerName)
            cols = ['CustomerID', 'Spending_Score', 'Churn_Risk']
            if 'CustomerName' in df.columns:
                cols.insert(1, 'CustomerName')
            churn_results = df[cols].copy()
            churn_results.columns = churn_results.columns.str.replace('_', ' ')

            # Save CSV
            output_file = f"churn_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
            churn_results.to_csv(output_path, index=False)
            download_link = url_for('download_results', filename=output_file)

        else:
            error = "CSV must contain 'CustomerID' and 'Spending_Score' columns."
    else:
        error = "No data available. Please upload a CSV file first."

    return render_template(
        'churn.html',
        error=error,
        churn_pie_chart=churn_pie_chart,
        churn_bar_chart=churn_bar_chart,
        churn_line_chart=churn_line_chart,
        churn_summary=churn_summary,
        churn_results=churn_results,
        download_link=download_link
    )
@app.route('/geo')
def geo_analysis():
    error = None
    geo_pie_chart = None
    geo_bar_chart = None
    geo_line_chart = None
    geo_summary = None
    geo_results = None

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]

        if 'Spending_Score' in df.columns and 'Region' in df.columns:
            df['Region'] = df['Region'].astype(str)

            geo_results = df.groupby('Region').agg({
                'CustomerID': 'count',
                'Spending_Score': 'mean'
            }).reset_index()
            geo_results.columns = ['Region', 'Customer Count', 'Average Spending Score']
            geo_results['Average Spending Score'] = geo_results['Average Spending Score'].round(2)

            pie_data = {
                'data': [{
                    'labels': geo_results['Region'].tolist(),
                    'values': geo_results['Customer Count'].tolist(),
                    'type': 'pie'
                }],
                'layout': {'title': 'Customer Distribution by Region'}
            }
            geo_pie_chart = f'''<div id="geo-pie"></div>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>Plotly.newPlot("geo-pie", {json.dumps(pie_data)});</script>'''

            bar_data = {
                'data': [{
                    'x': geo_results['Region'].tolist(),
                    'y': geo_results['Average Spending Score'].tolist(),
                    'type': 'bar',
                    'marker': {'color': 'orange'}
                }],
                'layout': {'title': 'Average Spending Score by Region', 'yaxis': {'title': 'Avg Spending'}}
            }
            geo_bar_chart = f'''<div id="geo-bar"></div>
            <script>Plotly.newPlot("geo-bar", {json.dumps(bar_data)});</script>'''

            if 'OrderDate' in df.columns:
                df['OrderDate'] = pd.to_datetime(df['OrderDate'])
                df['Month'] = df['OrderDate'].dt.to_period('M').astype(str)

                monthly = df.groupby('Month')['Spending_Score'].mean().reset_index()
                line_data = {
                    'data': [{
                        'x': monthly['Month'].tolist(),
                        'y': monthly['Spending_Score'].round(2).tolist(),
                        'mode': 'lines+markers',
                        'line': {'color': 'blue'}
                    }],
                    'layout': {'title': 'Average Spending Over Time', 'xaxis': {'title': 'Month'}, 'yaxis': {'title': 'Avg Spending'}}
                }
                geo_line_chart = f'''<div id="geo-line"></div>
                <script>Plotly.newPlot("geo-line", {json.dumps(line_data)});</script>'''

            geo_summary = {
                "Total Regions": df['Region'].nunique(),
                "Total Customers": df['CustomerID'].nunique(),
                "Overall Avg Spending Score": f"{df['Spending_Score'].mean():.2f}",
                "Max Region Spending Avg": f"{geo_results['Average Spending Score'].max():.2f}",
                "Min Region Spending Avg": f"{geo_results['Average Spending Score'].min():.2f}"
            }

        else:
            error = "Uploaded CSV must contain 'Spending_Score' and 'Region' columns."
    else:
        error = "No data available. Please upload a valid CSV."

    return render_template('geo.html',
        error=error,
        geo_pie_chart=geo_pie_chart,
        geo_bar_chart=geo_bar_chart,
        geo_line_chart=geo_line_chart,
        geo_summary=geo_summary,
        geo_results=geo_results
    )

@app.route('/trends')
def trends_analysis():
    error = None
    line_chart = None
    bar_chart = None
    pie_chart = None
    trends_results = pd.DataFrame()
    download_link = None
    selected_month = request.args.get('month', 'All')

    file_id = session.get('last_uploaded_file_id')
    if file_id and file_id in uploaded_data:
        df = uploaded_data[file_id]

        # Ensure necessary columns exist
        if 'Spending_Score' in df.columns and ('Date' in df.columns or 'PurchaseDate' in df.columns):
            time_column = 'Date' if 'Date' in df.columns else 'PurchaseDate'
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            df.dropna(subset=[time_column], inplace=True)
            df['Month'] = df[time_column].dt.to_period('M').astype(str)

            # Filter by selected month
            if selected_month != 'All':
                df = df[df['Month'] == selected_month]

            # Grouped results
            trends_results = df.groupby('Month').agg({
                'CustomerID': 'count',
                'Spending_Score': 'mean'
            }).reset_index()
            trends_results.columns = ['Month', 'Customer Count', 'Average Spending Score']
            trends_results['Average Spending Score'] = trends_results['Average Spending Score'].round(2)

            # Pie Chart
            pie_data = {
                'data': [{
                    'labels': trends_results['Month'].tolist(),
                    'values': trends_results['Customer Count'].tolist(),
                    'type': 'pie'
                }],
                'layout': {'title': 'Customer Purchase Share by Month'}
            }
            pie_chart = f'''
                <div id="pie-chart"></div>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>Plotly.newPlot("pie-chart", {json.dumps(pie_data)});</script>
            '''

            # Line Chart
            line_data = {
                'data': [{
                    'x': trends_results['Month'].tolist(),
                    'y': trends_results['Average Spending Score'].tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Avg Spending'
                }],
                'layout': {
                    'title': 'Purchase Trends Over Time',
                    'xaxis': {'title': 'Month'},
                    'yaxis': {'title': 'Avg Spending Score'}
                }
            }
            line_chart = f'''
                <div id="line-chart"></div>
                <script>Plotly.newPlot("line-chart", {json.dumps(line_data)});</script>
            '''

            # Bar Chart
            bar_data = {
                'data': [{
                    'x': trends_results['Month'].tolist(),
                    'y': trends_results['Customer Count'].tolist(),
                    'type': 'bar',
                    'name': 'Customer Count'
                }],
                'layout': {
                    'title': 'Customer Count by Month',
                    'xaxis': {'title': 'Month'},
                    'yaxis': {'title': 'Count'}
                }
            }
            bar_chart = f'''
                <div id="bar-chart"></div>
                <script>Plotly.newPlot("bar-chart", {json.dumps(bar_data)});</script>
            '''

            # Export to CSV and link for download
            output_file = f"trend_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
            trends_results.to_csv(output_path, index=False)
            download_link = url_for('download_trends', filename=output_file)

        else:
            error = "CSV must contain 'Spending_Score' and either 'Date' or 'PurchaseDate'."
    else:
        error = "No data available. Please upload a CSV file first."

    return render_template(
        'trends.html',
        error=error,
        line_chart=line_chart,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        trends_results=trends_results,
        download_link=download_link,
        selected_month=selected_month
    )

@app.route('/download_trends/<filename>')
def download_trends(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return f"File {filename} not found.", 404

@app.route('/predictive')
def predictive_analysis():
    error = None
    line_chart = None
    bar_chart = None
    pie_chart = None
    predictive_analysis_results = pd.DataFrame()
    output_file = None
    download_link = None
    month_list = []
    selected_month = request.args.get('month', 'All')

    if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
        df = uploaded_data[session['last_uploaded_file_id']]

        if 'Spending_Score' in df.columns and ('Date' in df.columns or 'PurchaseDate' in df.columns):
            time_column = 'Date' if 'Date' in df.columns else 'PurchaseDate'
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
            df.dropna(subset=[time_column], inplace=True)
            df['Month'] = df[time_column].dt.to_period('M').astype(str)

            # Save full month list for dropdown
            month_list = df['Month'].unique().tolist()

            # Filter if needed
            if selected_month != 'All':
                df = df[df['Month'] == selected_month]

            # Group & Predict
            monthly_avg = df.groupby('Month')['Spending_Score'].mean().sort_index().reset_index()

            if not monthly_avg.empty:
                last_month = monthly_avg['Month'].iloc[-1]
                next_month = (pd.Period(last_month, freq='M') + 1).strftime('%Y-%m')
                predicted_value = monthly_avg['Spending_Score'].iloc[-1] * 1.1

                predictive_analysis_results = pd.DataFrame({
                    'Month': [next_month],
                    'Predicted Average Spending Score': [round(predicted_value, 2)]
                })

                months = monthly_avg['Month'].tolist() + [next_month]
                scores = monthly_avg['Spending_Score'].tolist() + [predicted_value]

                # Pie chart (exclude prediction)
                pie_data = {
                    'data': [{
                        'labels': monthly_avg['Month'].tolist(),
                        'values': monthly_avg['Spending_Score'].tolist(),
                        'type': 'pie'
                    }],
                    'layout': {'title': 'Spending Share by Month'}
                }
                pie_chart = f'''<div id="pie-chart"></div>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>Plotly.newPlot("pie-chart", {json.dumps(pie_data)});</script>'''

                # Line chart
                line_data = {
                    'data': [{
                        'x': months,
                        'y': scores,
                        'type': 'scatter',
                        'mode': 'lines+markers',
                        'name': 'Predicted Spending'
                    }],
                    'layout': {
                        'title': 'Predicted Trends Over Time',
                        'xaxis': {'title': 'Month'},
                        'yaxis': {'title': 'Avg Spending Score'}
                    }
                }
                line_chart = f'''
                <div id="line-chart"></div>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>Plotly.newPlot("line-chart", {json.dumps(line_data)});</script>'''

                # Bar chart
                bar_data = {
                    'data': [{
                        'x': months,
                        'y': scores,
                        'type': 'bar',
                        'name': 'Predicted Spending'
                    }],
                    'layout': {
                        'title': 'Predicted Monetary Value by Period',
                        'xaxis': {'title': 'Month'},
                        'yaxis': {'title': 'Avg Spending Score'}
                    }
                }
                bar_chart = f'''
                <div id="bar-chart"></div>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>Plotly.newPlot("bar-chart", {json.dumps(bar_data)});</script>'''

                # CSV export
                output_file = f"predictive_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
                path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
                predictive_analysis_results.to_csv(path, index=False)
                download_link = url_for('download_predictive', filename=output_file)
        else:
            error = "Uploaded CSV must contain 'Spending_Score' and either 'Date' or 'PurchaseDate'."
    else:
        error = "No data available. Please upload a valid CSV."

    return render_template(
        'predictive_analysis.html',
        error=error,
        line_chart=line_chart,
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        predictive_analysis_results=predictive_analysis_results,
        download_link=download_link,
        selected_month=selected_month,
        month_list=month_list
    )
@app.route('/download_predictive/<filename>')
def download_predictive(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return f"File {filename} not found.", 404

@app.route('/results')
def results():
    error = None
    detailed_results = None
    results_summary = None
    combined_charts = {}
    output_file = None
    selected_month = request.args.get('month', 'All')
    month_list = []

    try:
        if 'last_uploaded_file_id' in session and session['last_uploaded_file_id'] in uploaded_data:
            df = uploaded_data[session['last_uploaded_file_id']]

            # Determine time column (either 'Date' or 'PurchaseDate')
            time_column = None
            if 'Date' in df.columns:
                time_column = 'Date'
            elif 'PurchaseDate' in df.columns:
                time_column = 'PurchaseDate'

            # Filter and parse dates
            if time_column:
                df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
                df.dropna(subset=[time_column], inplace=True)
                df['Month'] = df[time_column].dt.to_period('M').astype(str)
                month_list = df['Month'].unique().tolist()

                if selected_month != 'All':
                    df = df[df['Month'] == selected_month]
            else:
                error = "Missing 'Date' or 'PurchaseDate' column for month filtering."

            # Proceed only if 'CustomerID' exists
            if 'CustomerID' in df.columns:
                if 'Spending_Score' in df.columns and 'Annual_Income' in df.columns:
                    detailed_results = df[['CustomerID', 'Spending_Score', 'Annual_Income']]
                else:
                    detailed_results = df[['CustomerID']]

                total_customers = len(df)
                avg_spending = df['Spending_Score'].mean() if 'Spending_Score' in df.columns else 0
                avg_income = df['Annual_Income'].mean() if 'Annual_Income' in df.columns else 0

                results_summary = {
                    "total_customers": total_customers,
                    "avg_spending": f"{avg_spending:.2f}",
                    "avg_income": f"{avg_income:.2f}"
                }


                # Pie chart by Segment
                if 'Segment' in df.columns:
                    pie_fig = px.pie(df, names='Segment', title='Customer Segment Distribution')
                    combined_charts['Customer Segment Distribution'] = to_html(pie_fig, full_html=False)

                # Line chart: Spending Score Over Time
                if time_column and 'Spending_Score' in df.columns:
                    df_sorted = df.sort_values(time_column)
                    if not df_sorted[time_column].isnull().all():
                        line_fig = px.line(df_sorted, x=time_column, y='Spending_Score', title='Spending Score Over Time')
                        combined_charts['Spending Score Over Time'] = to_html(line_fig, full_html=False)

                # Bar chart: Avg Income by Segment
                if 'Annual_Income' in df.columns and 'Segment' in df.columns:
                    bar_df = df.groupby('Segment')['Annual_Income'].mean().reset_index()
                    bar_fig = px.bar(bar_df, x='Segment', y='Annual_Income', title='Average Income by Segment')
                    combined_charts['Average Income by Segment'] = to_html(bar_fig, full_html=False)

            else:
                error = "Uploaded CSV must contain a 'CustomerID' column."
        else:
            error = "No data available. Please upload a valid CSV first."

    except Exception as e:
        error = f"An error occurred during processing: {str(e)}"

    return render_template(
        'results.html',
        error=error,
        results_summary=results_summary,
        detailed_results=detailed_results,
        combined_charts=combined_charts,
        output_file=output_file,
        selected_month=selected_month,
        month_list=month_list
    )


@app.route('/download_results/<filename>')
def download_results(filename):
    try:
        return send_from_directory(
            directory=app.config['OUTPUT_FOLDER'],
            path=filename,
            as_attachment=True
        )
    except FileNotFoundError:
        return "File not found", 404



if __name__ == '__main__':
    app.run(debug=True)