import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import json
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import base64
from email.mime.text import MIMEText
import html
from urllib.parse import urlencode

# Page config
st.set_page_config(
    page_title="Bank Transaction Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Categories with colors
CATEGORIES = {
    'Food & Dining': {
        'color': '#FF6B6B',
        'subcategories': {
            'Restaurants': '#FF4757',
            'Fast Food': '#FF3838',
            'Groceries': '#FF6B6B'
        }
    },
    'Entertainment': {
        'color': '#4ECDC4',
        'subcategories': {
            'Netflix': '#E50914',
            'Amazon Prime': '#FF9900',
            'Movies': '#4ECDC4'
        }
    },
    'Shopping': {
        'color': '#45B7D1',
        'subcategories': {
            'Clothing': '#45B7D1',
            'Electronics': '#3498DB',
            'General': '#5DADE2'
        }
    },
    'Transportation': {
        'color': '#FFA07A',
        'subcategories': {
            'Fuel': '#FF7F50',
            'Public Transport': '#FFA07A',
            'Taxi/Ride Share': '#FF8C69'
        }
    },
    'Bills & Utilities': {
        'color': '#98D8C8',
        'subcategories': {
            'Electricity': '#98D8C8',
            'Internet': '#7FCDCD',
            'Phone': '#66CDAA'
        }
    }
}

# Bank email patterns
BANK_PATTERNS = {
    'sbi': {
        'sender': 'donotreply.sbiatm@alerts.sbi.co.in',
        'amount_pattern': r'Amount \(INR\)\s*([0-9,]+\.?[0-9]*)',
        'merchant_pattern': r'Terminal Owner Name\s*([^\n\r]+)',
        'date_pattern': r'Date & Time\s*([^\n\r]+)',
        'card_pattern': r'Last 4 Digit of Card\s*([^\n\r]+)'
    },
    'hdfc': {
        'sender': 'alerts@hdfcbank.net',
        'amount_pattern': r'Rs\.?\s*([0-9,]+\.?[0-9]*)',
        'merchant_pattern': r'at\s+([^\s]+)',
        'date_pattern': r'on\s+([0-9]{2}-[0-9]{2}-[0-9]{4})',
        'card_pattern': r'card\s+ending\s+([0-9]{4})'
    },
    'icici': {
        'sender': ['alert@icicibank.com', 'credit_cards@icicibank.com'],
        'amount_pattern': r'INR\s*([0-9,]+\.?[0-9]*)',
        'merchant_pattern': r'at\s+([^\n\r]+)',
        'date_pattern': r'on\s+([0-9]{2}-[0-9]{2}-[0-9]{4})',
        'card_pattern': r'Card\s+ending\s+([0-9]{4})'
    }
}

class BankEmailAnalyzer:
    def __init__(self):
        self.replicate_token = st.secrets.get("REPLICATE_API_TOKEN", "")
        
    def parse_transaction_email(self, email_content, sender):
        """Parse transaction details from email content"""
        try:
            # Determine bank from sender
            bank = self.identify_bank(sender)
            if not bank:
                return None
                
            patterns = BANK_PATTERNS.get(bank, {})
            
            # Extract amount
            amount_match = re.search(patterns.get('amount_pattern', ''), email_content, re.IGNORECASE)
            amount = float(amount_match.group(1).replace(',', '')) if amount_match else 0.0
            
            # Extract merchant
            merchant_match = re.search(patterns.get('merchant_pattern', ''), email_content, re.IGNORECASE)
            merchant = merchant_match.group(1).strip() if merchant_match else "Unknown"
            
            # Extract date
            date_match = re.search(patterns.get('date_pattern', ''), email_content, re.IGNORECASE)
            date_str = date_match.group(1).strip() if date_match else ""
            
            # Extract card info
            card_match = re.search(patterns.get('card_pattern', ''), email_content, re.IGNORECASE)
            card_last4 = card_match.group(1).strip() if card_match else ""
            
            return {
                'amount': amount,
                'merchant': merchant,
                'date': self.parse_date(date_str),
                'card_last4': card_last4,
                'bank': bank.upper(),
                'raw_email': email_content
            }
        except Exception as e:
            st.error(f"Error parsing email: {str(e)}")
            return None
    
    def identify_bank(self, sender):
        """Identify bank from sender email"""
        sender_lower = sender.lower()
        if 'sbi' in sender_lower:
            return 'sbi'
        elif 'hdfc' in sender_lower:
            return 'hdfc'
        elif 'icici' in sender_lower:
            return 'icici'
        elif 'axis' in sender_lower:
            return 'axis'
        elif 'kotak' in sender_lower:
            return 'kotak'
        elif 'idfc' in sender_lower:
            return 'idfc'
        elif 'yes' in sender_lower:
            return 'yes'
        elif 'indus' in sender_lower:
            return 'indus'
        return None
    
    def parse_date(self, date_str):
        """Parse date from various formats"""
        try:
            # Try different date formats
            formats = [
                '%b %d, %Y, %H:%M',
                '%d-%m-%Y',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return datetime.now()
        except:
            return datetime.now()
    
    def categorize_transaction_ai(self, merchant, amount):
        """Use AI to categorize transaction"""
        if not self.replicate_token:
            return self.fallback_categorization(merchant)
        
        try:
            prompt = f"""
            Categorize this transaction:
            Merchant: {merchant}
            Amount: ₹{amount}
            
            Choose from these categories:
            - Food & Dining (Restaurants, Fast Food, Groceries)
            - Entertainment (Netflix, Amazon Prime, Movies)
            - Shopping (Clothing, Electronics, General)
            - Transportation (Fuel, Public Transport, Taxi/Ride Share)
            - Bills & Utilities (Electricity, Internet, Phone)
            
            Respond with only: Category|Subcategory
            Example: Food & Dining|Restaurants
            """
            
            response = requests.post(
                "https://api.replicate.com/v1/models/openai/gpt-4o-mini/predictions",
                headers={
                    "Authorization": f"Bearer {self.replicate_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": {
                        "prompt": prompt,
                        "system_prompt": "You are a financial categorization expert. Respond only with Category|Subcategory format."
                    }
                }
            )
            
            if response.status_code == 201:
                prediction_url = response.json()["urls"]["get"]
                
                # Poll for result
                for _ in range(10):
                    result_response = requests.get(
                        prediction_url,
                        headers={"Authorization": f"Bearer {self.replicate_token}"}
                    )
                    
                    if result_response.status_code == 200:
                        result = result_response.json()
                        if result["status"] == "succeeded":
                            output = result["output"]
                            if isinstance(output, list):
                                output = "".join(output)
                            
                            if "|" in output:
                                category, subcategory = output.strip().split("|", 1)
                                return category.strip(), subcategory.strip()
                            break
                    
                    import time
                    time.sleep(1)
            
        except Exception as e:
            st.warning(f"AI categorization failed: {str(e)}")
        
        return self.fallback_categorization(merchant)
    
    def fallback_categorization(self, merchant):
        """Fallback categorization based on merchant keywords"""
        merchant_lower = merchant.lower()
        
        # Food & Dining
        if any(word in merchant_lower for word in ['restaurant', 'food', 'cafe', 'pizza', 'burger', 'swiggy', 'zomato', 'dominos', 'mcd', 'kfc']):
            if any(word in merchant_lower for word in ['swiggy', 'zomato', 'delivery']):
                return 'Food & Dining', 'Fast Food'
            return 'Food & Dining', 'Restaurants'
        
        # Shopping
        if any(word in merchant_lower for word in ['amazon', 'flipkart', 'myntra', 'reliance', 'retail', 'store', 'mall']):
            return 'Shopping', 'General'
        
        # Entertainment
        if any(word in merchant_lower for word in ['netflix', 'prime', 'hotstar', 'cinema', 'movie', 'theatre']):
            if 'netflix' in merchant_lower:
                return 'Entertainment', 'Netflix'
            elif 'prime' in merchant_lower:
                return 'Entertainment', 'Amazon Prime'
            return 'Entertainment', 'Movies'
        
        # Transportation
        if any(word in merchant_lower for word in ['petrol', 'fuel', 'gas', 'uber', 'ola', 'metro', 'bus']):
            if any(word in merchant_lower for word in ['petrol', 'fuel', 'gas']):
                return 'Transportation', 'Fuel'
            return 'Transportation', 'Taxi/Ride Share'
        
        # Default
        return 'Shopping', 'General'

class GmailAuth:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", "")
        self.CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
        self.REDIRECT_URI = st.secrets.get("REDIRECT_URI", "https://localhost:8501")
    
    def get_auth_url(self):
        """Get OAuth authorization URL"""
        try:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.CLIENT_ID,
                        "client_secret": self.CLIENT_SECRET,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [self.REDIRECT_URI]
                    }
                },
                scopes=self.SCOPES
            )
            flow.redirect_uri = self.REDIRECT_URI
            
            auth_url, _ = flow.authorization_url(prompt='consent')
            return auth_url
        except Exception as e:
            st.error(f"Error generating auth URL: {str(e)}")
            return None
    
    def get_emails(self, creds, query="", max_results=50):
        """Fetch emails from Gmail"""
        try:
            service = build('gmail', 'v1', credentials=creds)
            
            # Search for bank transaction emails
            bank_senders = [
                'donotreply.sbiatm@alerts.sbi.co.in',
                'alerts@hdfcbank.net',
                'alert@icicibank.com',
                'credit_cards@icicibank.com',
                'alerts@axisbank.com',
                'creditcardalerts@kotak.com',
                'noreply@idfcfirstbank.com',
                'alerts@yesbank.in',
                'transactionalert@indusind.com'
            ]
            
            search_query = f"from:({' OR '.join(bank_senders)})"
            if query:
                search_query += f" {query}"
            
            results = service.users().messages().list(
                userId='me',
                q=search_query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                msg = service.users().messages().get(
                    userId='me',
                    id=message['id']
                ).execute()
                
                payload = msg['payload']
                headers = payload.get('headers', [])
                
                # Extract email metadata
                sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
                date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                
                # Extract email body
                body = self.extract_email_body(payload)
                
                emails.append({
                    'id': message['id'],
                    'sender': sender,
                    'subject': subject,
                    'date': date,
                    'body': body
                })
            
            return emails
        except Exception as e:
            st.error(f"Error fetching emails: {str(e)}")
            return []
    
    def extract_email_body(self, payload):
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif part['mimeType'] == 'text/html' and 'data' in part['body']:
                    html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    body += html.unescape(re.sub('<[^<]+?>', '', html_content))
        elif payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body

def main():
    st.title("💳 Bank Transaction Email Analyzer")
    st.markdown("---")
    
    # Initialize classes
    analyzer = BankEmailAnalyzer()
    gmail_auth = GmailAuth()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        
        # Authentication status
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            st.warning("Please authenticate with Gmail")
            
            # Manual auth code input for Streamlit Cloud
            st.subheader("Gmail OAuth")
            auth_url = gmail_auth.get_auth_url()
            
            if auth_url:
                st.markdown(f"[🔗 Click here to authenticate]({auth_url})")
                
                auth_code = st.text_input("Enter authorization code:", type="password")
                
                if st.button("Authenticate") and auth_code:
                    try:
                        # Exchange code for tokens
                        flow = Flow.from_client_config(
                            {
                                "web": {
                                    "client_id": gmail_auth.CLIENT_ID,
                                    "client_secret": gmail_auth.CLIENT_SECRET,
                                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                    "token_uri": "https://oauth2.googleapis.com/token",
                                    "redirect_uris": [gmail_auth.REDIRECT_URI]
                                }
                            },
                            scopes=gmail_auth.SCOPES
                        )
                        flow.redirect_uri = gmail_auth.REDIRECT_URI
                        flow.fetch_token(code=auth_code)
                        
                        st.session_state.credentials = flow.credentials
                        st.session_state.authenticated = True
                        st.success("✅ Authenticated successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Authentication failed: {str(e)}")
        else:
            st.success("✅ Authenticated")
            
            # Date filter
            st.subheader("📅 Date Filter")
            date_range = st.date_input(
                "Select date range:",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                max_value=datetime.now()
            )
            
            # Fetch emails button
            if st.button("🔄 Fetch Transactions"):
                with st.spinner("Fetching and analyzing transactions..."):
                    emails = gmail_auth.get_emails(st.session_state.credentials)
                    
                    transactions = []
                    for email in emails:
                        transaction = analyzer.parse_transaction_email(email['body'], email['sender'])
                        if transaction and transaction['amount'] > 0:
                            # AI categorization
                            category, subcategory = analyzer.categorize_transaction_ai(
                                transaction['merchant'], 
                                transaction['amount']
                            )
                            
                            transaction['category'] = category
                            transaction['subcategory'] = subcategory
                            transactions.append(transaction)
                    
                    # Filter by date
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        transactions = [
                            t for t in transactions 
                            if start_date <= t['date'].date() <= end_date
                        ]
                    
                    st.session_state.transactions = transactions
                    st.success(f"✅ Analyzed {len(transactions)} transactions")
    
    # Main content
    if st.session_state.authenticated and 'transactions' in st.session_state:
        transactions = st.session_state.transactions
        
        if transactions:
            # Convert to DataFrame
            df = pd.DataFrame(transactions)
            df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(df))
            
            with col2:
                st.metric("Total Amount", f"₹{df['amount'].sum():,.2f}")
            
            with col3:
                st.metric("Average Transaction", f"₹{df['amount'].mean():,.2f}")
            
            with col4:
                unique_merchants = df['merchant'].nunique()
                st.metric("Unique Merchants", unique_merchants)
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Spending by Category")
                category_spending = df.groupby('category')['amount'].sum().reset_index()
                
                fig_pie = px.pie(
                    category_spending,
                    values='amount',
                    names='category',
                    color='category',
                    color_discrete_map={cat: CATEGORIES[cat]['color'] for cat in CATEGORIES.keys()},
                    title="Category Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("📈 Daily Spending Trend")
                daily_spending = df.groupby('date_str')['amount'].sum().reset_index()
                
                fig_line = px.line(
                    daily_spending,
                    x='date_str',
                    y='amount',
                    title="Daily Spending",
                    markers=True
                )
                fig_line.update_layout(xaxis_title="Date", yaxis_title="Amount (₹)")
                st.plotly_chart(fig_line, use_container_width=True)
            
            # Subcategory breakdown
            st.subheader("🏷️ Subcategory Breakdown")
            subcategory_spending = df.groupby(['category', 'subcategory'])['amount'].sum().reset_index()
            
            fig_bar = px.bar(
                subcategory_spending,
                x='subcategory',
                y='amount',
                color='category',
                title="Spending by Subcategory",
                color_discrete_map={cat: CATEGORIES[cat]['color'] for cat in CATEGORIES.keys()}
            )
            fig_bar.update_layout(xaxis_title="Subcategory", yaxis_title="Amount (₹)")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Transaction table
            st.subheader("📋 Transaction Details")
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_categories = st.multiselect(
                    "Filter by Category:",
                    options=df['category'].unique(),
                    default=df['category'].unique()
                )
            
            with col2:
                selected_banks = st.multiselect(
                    "Filter by Bank:",
                    options=df['bank'].unique(),
                    default=df['bank'].unique()
                )
            
            with col3:
                min_amount = st.number_input("Minimum Amount:", min_value=0.0, value=0.0)
            
            # Apply filters
            filtered_df = df[
                (df['category'].isin(selected_categories)) &
                (df['bank'].isin(selected_banks)) &
                (df['amount'] >= min_amount)
            ]
            
            # Display table
            display_df = filtered_df[['date_str', 'merchant', 'amount', 'category', 'subcategory', 'bank', 'card_last4']].copy()
            display_df.columns = ['Date', 'Merchant', 'Amount (₹)', 'Category', 'Subcategory', 'Bank', 'Card']
            display_df = display_df.sort_values('Date', ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Export functionality
            if st.button("📥 Export to CSV"):
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No transactions found for the selected date range.")
    
    elif not st.session_state.authenticated:
        st.info("👈 Please authenticate with Gmail to start analyzing your transactions.")
        
        # Demo section
        st.subheader("🎯 Features")
        st.markdown("""
        - **Gmail OAuth Integration**: Securely connect your Gmail account
        - **Multi-Bank Support**: Supports SBI, HDFC, ICICI, Axis, Kotak, IDFC, Yes Bank, IndusInd
        - **AI-Powered Categorization**: Automatically categorize transactions using AI
        - **Interactive Charts**: Visualize spending patterns and trends
        - **Date Filtering**: Analyze transactions for specific time periods
        - **Export Functionality**: Download transaction data as CSV
        - **Responsive Design**: Works on desktop and mobile devices
        """)
        
        # Sample transaction display
        st.subheader("📧 Sample Transaction Email")
        sample_email = """
        **Dear Valued SBI Debit Card Holder,**
        
        The below transaction has been done using your SBI debit card.
        
        Description | Value
        Terminal Owner Name | RELIANCE RETAIL LTD
        Terminal Id | 89051784
        Date & Time | Jun 21, 2025, 16:10
        Transaction Number | 517210057033
        Amount (INR) | 349.00
        Last 4 Digit of Card | X3093
        Transaction Type | PURCHASE
        Channel | POS / ECOM
        City | 01204770770
        Location | RELIANCE RETAIL LTD
        """
        st.code(sample_email)
    
    else:
        st.info("Click 'Fetch Transactions' in the sidebar to analyze your emails.")

if __name__ == "__main__":
    main()
