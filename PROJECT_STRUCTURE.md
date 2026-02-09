# Project Structure - Backend & Frontend Organization

## 📁 Directory Layout

```
PROJECT AI/
│
├── backend/                          # ⚙️ Server-side logic & data
│   ├── app.py                        # Main Flask application
│   ├── demographic.py                # Demographic trend calculations
│   ├── users.json                    # User authentication data
│   ├── .env                          # Environment variables (API keys, secrets)
│   ├── .env.secret                   # Additional secret configuration
│   ├── uploads/                      # User-uploaded CSV files
│   ├── outputs/                      # Generated analysis results
│   └── __pycache__/                  # Python compiled files
│
├── frontend/                         # 🎨 User interface & styling
│   ├── templates/                    # HTML pages (Flask Jinja2)
│   │   ├── base.html                 # Base layout with chat bubble
│   │   ├── index.html                # Home/upload page
│   │   ├── login.html                # Modern login page
│   │   ├── register.html             # Modern registration page
│   │   ├── dashboard.html            # User dashboard
│   │   ├── ask_data.html             # Natural language query interface
│   │   ├── overview.html             # Data overview
│   │   ├── demographic.html          # Demographic trends analysis
│   │   ├── rfm.html                  # RFM market analysis
│   │   ├── behavior.html             # Customer behavior analysis
│   │   ├── product_preference.html   # Product preference analysis
│   │   ├── churn.html                # Churn analysis
│   │   ├── clv.html                  # Customer Lifetime Value
│   │   ├── ceo_business_strategies.html
│   │   ├── ceo_financial_trends.html
│   │   ├── ceo_performance_metrics.html
│   │   └── error.html                # Error page template
│   │
│   └── static/                       # CSS, JavaScript, assets
│       ├── style.css                 # Main stylesheet
│       ├── chat.js                   # Floating chat bubble logic
│       ├── rfm_output.csv            # Generated RFM analysis export
│       └── ...
│
├── .venv/                            # Python virtual environment
├── .git/                             # Git repository
├── .gitignore                        # Git ignore rules
│
├── README.md                         # Project documentation
├── PROJECT_STRUCTURE.md              # This file
├── QUICK_START_GUIDE.md              # How to run the project
├── IMPLEMENTATION_SUMMARY.md         # Feature summary
├── CHAT_BUBBLE_DOCUMENTATION.md      # Chat bubble feature guide
└── DATA_INTERFACE_DOCUMENTATION.md   # API documentation

```

## 🚀 How to Run

### 1. Navigate to Backend
```bash
cd backend
```

### 2. Activate Virtual Environment
```bash
.\.venv\Scripts\Activate.ps1  # PowerShell (Windows)
# or
source .venv/bin/activate      # bash/zsh (Mac/Linux)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Flask Application
```bash
python app.py
```

The app will start on `http://localhost:5000`

---

## 📊 Technology Stack

### **Backend**
- **Framework**: Flask (Python)
- **AI/LLM**: Google Gemini 1.5 Flash via LangChain
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Memory**: LangChain ConversationBufferMemory
- **Database**: JSON (users.json), MongoDB (optional)

### **Frontend**
- **HTML**: Jinja2 templates (Flask)
- **CSS**: Bootstrap 5, Custom styles
- **JavaScript**: AJAX, Plotly.js
- **Fonts**: Inter, Segoe UI
- **UI Features**: Floating chat bubble, responsive design

---

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `backend/app.py` | Main Flask application with all routes and AI logic |
| `backend/demographic.py` | Demographic trend analysis calculations |
| `backend/users.json` | User credentials and roles (hashed passwords) |
| `frontend/templates/base.html` | Base layout + floating chat bubble |
| `frontend/templates/login.html` | Modern authentication page |
| `frontend/templates/dashboard.html` | User dashboard with strategy buttons |
| `frontend/static/chat.js` | Chat bubble AJAX communication |
| `backend/.env` | Environment variables (Google API key) |

---

## 🔐 Security Features

- ✅ Hashed password storage (werkzeug.security)
- ✅ Session-based authentication
- ✅ Role-based access control (CEO / Company Analyst)
- ✅ SQL injection prevention (read-only dataframe queries)
- ✅ Dangerous keyword filtering (13+ patterns blocked)
- ✅ Prompt engineering for safe AI responses
- ✅ XSS protection in chat messages

---

## 📝 Configuration

Update `backend/.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
SECRET_KEY=your_flask_secret_key
MONGODB_URI=mongodb://connection_string  # Optional
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
USERS_FILE=users.json
```

---

## 🎯 Main Features

1. **AI-Powered Chat Bubble** - Float on all pages, natural language data queries
2. **Customer Analytics** - RFM segmentation, CLV analysis, churn prediction
3. **Dynamic Strategies** - Generate business strategies based on data
4. **Multi-turn Conversations** - Chat history preserved per user
5. **Regional Filtering** - Analyze data by East African cities
6. **Modern UI** - Premium design with gradient backgrounds, card layouts
7. **Read-Only Access** - Safe, non-destructive data analysis
8. **Responsive Design** - Works on desktop and mobile

---

## 📞 Support

For issues, refer to:
- `QUICK_START_GUIDE.md` - Getting started
- `IMPLEMENTATION_SUMMARY.md` - Feature overview
- `CHAT_BUBBLE_DOCUMENTATION.md` - Chat integration
- `DATA_INTERFACE_DOCUMENTATION.md` - API details

---

**Last Updated**: February 2, 2026
