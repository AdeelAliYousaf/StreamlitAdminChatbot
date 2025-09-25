import os
import re
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dataclasses import dataclass

# Configuration
@dataclass
class Config:
    DB_PATH = "users.db"
    MODEL_NAME = "all-MiniLM-L6-v2"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
genai.configure(api_key=Config.GEMINI_API_KEY)

class UserDatabase:
    """Database handler for user management"""
    
    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with users table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            phone TEXT,
            city TEXT,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create admin emails table for auto-login
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Insert default admin if not exists
        cursor.execute('INSERT OR IGNORE INTO admin_emails (email) VALUES (?)', 
                      ('admin@company.com',))
        
        conn.commit()
        conn.close()
    
    def is_admin(self, email: str) -> bool:
        """Check if email is in admin list"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM admin_emails WHERE email = ?', (email,))
        result = cursor.fetchone()[0] > 0
        conn.close()
        return result
    
    def add_user(self, email: str, name: str = None, phone: str = None, 
                 city: str = None, department: str = None) -> bool:
        """Add a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO users (email, name, phone, city, department, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (email, name, phone, city, department))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def update_user(self, email: str = None, name: str = None, **kwargs) -> bool:
        """Update user information"""
        if not email and not name:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find user by email or name
        if email:
            cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        else:
            cursor.execute('SELECT id FROM users WHERE name LIKE ?', (f'%{name}%',))
        
        user_row = cursor.fetchone()
        if not user_row:
            conn.close()
            return False
        
        user_id = user_row[0]
        
        # Update fields
        update_fields = []
        update_values = []
        
        for field, value in kwargs.items():
            if field in ['phone', 'city', 'department', 'name'] and value:
                update_fields.append(f'{field} = ?')
                update_values.append(value)
        
        if update_fields:
            update_fields.append('updated_at = CURRENT_TIMESTAMP')
            update_values.append(user_id)
            
            query = f'UPDATE users SET {", ".join(update_fields)} WHERE id = ?'
            cursor.execute(query, update_values)
            conn.commit()
        
        conn.close()
        return True
    
    def delete_user(self, email: str = None, name: str = None) -> bool:
        """Delete a user"""
        if not email and not name:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if email:
            cursor.execute('DELETE FROM users WHERE email = ?', (email,))
        else:
            cursor.execute('DELETE FROM users WHERE name LIKE ?', (f'%{name}%',))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
    def search_users(self, query: str) -> List[Dict]:
        """Search users by various fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        search_pattern = f'%{query}%'
        cursor.execute('''
        SELECT email, name, phone, city, department, created_at, updated_at
        FROM users
        WHERE email LIKE ? OR name LIKE ? OR city LIKE ? OR department LIKE ?
        ORDER BY updated_at DESC
        ''', (search_pattern, search_pattern, search_pattern, search_pattern))
        
        columns = ['email', 'name', 'phone', 'city', 'department', 'created_at', 'updated_at']
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_all_users(self) -> List[Dict]:
        """Get all users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT email, name, phone, city, department, created_at, updated_at
        FROM users
        ORDER BY updated_at DESC
        ''')
        
        columns = ['email', 'name', 'phone', 'city', 'department', 'created_at', 'updated_at']
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class CommandParser:
    """Natural language command parser for user management using Gemini AI"""
    
    def __init__(self):
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'[\+]?[0-9\-\(\)\s]+'
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
    def extract_with_gemini(self, text: str) -> Dict:
        """Use Gemini AI to extract structured information from natural language"""
        prompt = f"""
        Analyze the following user management command and extract structured information:
        
        Command: "{text}"
        
        Please extract the following information and return ONLY a JSON object:
        {{
            "action": "add/update/delete/search or null if unclear",
            "email": "extracted email or null",
            "name": "extracted name or null", 
            "phone": "extracted phone number or null",
            "city": "extracted city or null",
            "department": "extracted department or null"
        }}
        
        Rules:
        - For action: use "add" for adding users, "update" for modifying, "delete" for removing, "search" for finding
        - Extract exact email addresses
        - Extract phone numbers (including country codes)
        - Extract person names (not email usernames)
        - Extract city names
        - Extract department names
        - Return null for fields that cannot be determined
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            # Clean the response to get just the JSON
            response_text = response.text.strip()
            
            # Remove any markdown formatting
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
        except Exception as e:
            print(f"Gemini parsing error: {e}")
            # Fallback to regex parsing
            return self.parse_command_regex(text)
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email from text"""
        matches = re.findall(self.email_pattern, text.lower())
        return matches[0] if matches else None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text"""
        # Look for phone patterns after keywords
        phone_keywords = ['phone', 'number', 'contact']
        for keyword in phone_keywords:
            pattern = f'{keyword}[:\s]+([+]?[0-9\-\(\)\s]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Look for quoted phone numbers
        quoted_phone = re.findall(r'["\']([+][0-9\-\(\)\s]+)["\']', text)
        if quoted_phone:
            return quoted_phone[0].strip()
        
        return None
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract name from text"""
        # Look for quoted names
        quoted_names = re.findall(r'["\']([A-Za-z\s]+)["\']', text)
        for name in quoted_names:
            if '@' not in name and not re.match(r'[+0-9\-\(\)\s]+', name):
                return name.strip()
        
        # Look for names after 'user'
        user_pattern = r'user\s+([A-Za-z\s]+?)(?:\s|$|[,.])'
        matches = re.findall(user_pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()
        
        return None
    
    def extract_city(self, text: str) -> Optional[str]:
        """Extract city from text"""
        city_pattern = r'city\s+(?:to\s+)?["\']?([A-Za-z\s]+)["\']?'
        matches = re.findall(city_pattern, text, re.IGNORECASE)
        return matches[0].strip() if matches else None
    
    def extract_department(self, text: str) -> Optional[str]:
        """Extract department from text"""
        dept_pattern = r'department\s+(?:to\s+)?["\']?([A-Za-z\s]+)["\']?'
        matches = re.findall(dept_pattern, text, re.IGNORECASE)
        return matches[0].strip() if matches else None
    
    def parse_command_regex(self, text: str) -> Dict:
        """Fallback regex-based command parsing"""
        text_lower = text.lower()
        
        command_info = {
            'action': None,
            'email': self.extract_email(text),
            'name': self.extract_name(text),
            'phone': self.extract_phone(text),
            'city': self.extract_city(text),
            'department': self.extract_department(text)
        }
        
        # Determine action
        if any(word in text_lower for word in ['add', 'create', 'insert', 'new']):
            command_info['action'] = 'add'
        elif any(word in text_lower for word in ['update', 'modify', 'change', 'edit']):
            command_info['action'] = 'update'
        elif any(word in text_lower for word in ['delete', 'remove', 'del']):
            command_info['action'] = 'delete'
        elif any(word in text_lower for word in ['search', 'find', 'look', 'show']):
            command_info['action'] = 'search'
        
        return command_info
    
    def parse_command(self, text: str) -> Dict:
        """Parse natural language command using Gemini AI with regex fallback"""
        return self.extract_with_gemini(text)

class RAGChatbot:
    """RAG-based chatbot for user management"""
    
    def __init__(self):
        self.db = UserDatabase()
        self.parser = CommandParser()
        self.encoder = SentenceTransformer(Config.MODEL_NAME)
        self.user_embeddings = None
        self.user_data = None
        self.update_embeddings()
    
    def update_embeddings(self):
        """Update user data embeddings for RAG"""
        users = self.db.get_all_users()
        if not users:
            self.user_embeddings = None
            self.user_data = None
            return
        
        # Create text representations of users
        user_texts = []
        for user in users:
            text_parts = []
            for key, value in user.items():
                if value and key != 'created_at' and key != 'updated_at':
                    text_parts.append(f"{key}: {value}")
            user_texts.append(" | ".join(text_parts))
        
        # Generate embeddings
        self.user_embeddings = self.encoder.encode(user_texts)
        self.user_data = users
    
    def find_relevant_users(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find relevant users using semantic search"""
        if self.user_embeddings is None:
            return []
        
        query_embedding = self.encoder.encode([query])
        similarities = cosine_similarity(query_embedding, self.user_embeddings)[0]
        
        # Get top-k most similar users
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_users = [self.user_data[i] for i in top_indices if similarities[i] > 0.3]
        
        return relevant_users
    
    def execute_command(self, command_info: Dict) -> Tuple[bool, str]:
        """Execute parsed command"""
        action = command_info['action']
        
        if action == 'add':
            if not command_info['email']:
                return False, "Email is required to add a user."
            
            success = self.db.add_user(
                email=command_info['email'],
                name=command_info['name'],
                phone=command_info['phone'],
                city=command_info['city'],
                department=command_info['department']
            )
            
            if success:
                self.update_embeddings()
                return True, f"Successfully added user {command_info['email']}"
            else:
                return False, f"User {command_info['email']} already exists"
        
        elif action == 'update':
            if not command_info['email'] and not command_info['name']:
                return False, "Email or name is required to update a user."
            
            update_data = {k: v for k, v in command_info.items() 
                          if k not in ['action', 'email'] and v is not None}
            
            success = self.db.update_user(
                email=command_info['email'],
                name=command_info['name'],
                **update_data
            )
            
            if success:
                self.update_embeddings()
                identifier = command_info['email'] or command_info['name']
                return True, f"Successfully updated user {identifier}"
            else:
                return False, "User not found"
        
        elif action == 'delete':
            if not command_info['email'] and not command_info['name']:
                return False, "Email or name is required to delete a user."
            
            success = self.db.delete_user(
                email=command_info['email'],
                name=command_info['name']
            )
    
            if success:
                self.update_embeddings()
                identifier = command_info['email'] or command_info['name']
                return True, f"Successfully deleted user {identifier}"
            else:
                return False, "User not found"
        
        elif action == 'search':
            query = command_info['email'] or command_info['name'] or ""
            users = self.find_relevant_users(query) if query else self.db.get_all_users()
            
            if users:
                return True, f"Found {len(users)} user(s)"
            else:
                return False, "No users found"
        
        return False, "Unknown command"
    
    def process_message(self, message: str) -> Tuple[str, List[Dict]]:
        """Process user message and return response"""
        # Parse command
        command_info = self.parser.parse_command(message)
        
        if command_info['action']:
            # Execute command
            success, response = self.execute_command(command_info)
            
            # For search commands, also return user data
            if command_info['action'] == 'search':
                query = command_info['email'] or command_info['name'] or message
                users = self.find_relevant_users(query) if query else self.db.get_all_users()
                return response, users
            else:
                return response, []
        
        else:
            # General query - use RAG to find relevant information
            relevant_users = self.find_relevant_users(message)
            
            if relevant_users:
                response = f"I found {len(relevant_users)} relevant user(s) based on your query."
                return response, relevant_users
            else:
                return "I didn't understand your command. Try commands like:\n- Add user john@example.com with phone +123456789\n- Update John's city to New York\n- Delete user john@example.com\n- Search for users in Marketing", []

# Streamlit Interface
def main():
    st.set_page_config(page_title="Adeel Ali Yousaf", page_icon="🤖", layout="wide")
    
    st.title("🤖 Test for AI Chatbot Development by WPBrigade")
    st.markdown("---")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    
    # Login section
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.subheader("🔐 Admin Login")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            email = st.text_input("Enter your admin email:", placeholder="admin@company.com")
        with col2:
            st.write("")
            st.write("")
            if st.button("Login", type="primary"):
                if st.session_state.chatbot.db.is_admin(email):
                    st.session_state.logged_in = True
                    st.session_state.admin_email = email
                    st.rerun()
                else:
                    st.error("Access denied. Email not in admin list.")
        
        st.info("💡 Default admin email: admin@company.com")
        return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "users" in message and message["users"]:
                    df = pd.DataFrame(message["users"])
                    st.dataframe(df, use_container_width=True)
        
        # Chat input
        if prompt := st.chat_input("Enter your command (e.g., 'add user john@example.com with phone +123456789')"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process message
            response, users = st.session_state.chatbot.process_message(prompt)
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "users": users
            })
            
            st.rerun()
    
    with col2:
        st.subheader("👤 Admin Panel")
        st.write(f"Logged in as: {st.session_state.admin_email}")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        all_users = st.session_state.chatbot.db.get_all_users()
        st.metric("Total Users", len(all_users))
        
        # Recent users
        st.subheader("📊 Recent Users")
        if all_users:
            recent_df = pd.DataFrame(all_users[:5])
            st.dataframe(recent_df[['email', 'name', 'city']], use_container_width=True)
        else:
            st.info("No users in database")
        
        st.markdown("---")
        
        # Command examples
        st.subheader("💡 Command Examples")
        examples = [
            "can you add the user john.smith@xyz.com with phone number +92332",
            "can you remove the user john.smith@xyz.com",
            "can you update  samanthas city to Cordoba"
        ]
        
        for example in examples:
            if st.button(f"📝 {example}", key=example):
                st.session_state.messages.append({"role": "user", "content": example})
                response, users = st.session_state.chatbot.process_message(example)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "users": users
                })
                st.rerun()

if __name__ == "__main__":
    main()