# RAG Admin Chatbot

An AI-powered **Admin Chatbot** built with **Streamlit**, **SQLite**, and **Google Gemini API**.  
It enables administrators to manage users (add, update, delete, search) using **natural language commands**, enhanced by **RAG (Retrieval Augmented Generation)** for semantic search.

---

## 🚀 Features
- 🔑 **Admin Authentication** – Only registered admins can access the chatbot.  
- 💬 **Natural Language Commands** – Add, update, delete, or search users via plain English.  
- 🗄️ **SQLite Database** – Stores users and admin data securely.  
- 🧠 **RAG + SentenceTransformer** – Finds relevant users using semantic similarity.  
- ⚡ **Google Gemini API** – Parses and understands user queries more accurately.  
- 📊 **Streamlit Admin Panel** – View, search, and manage users with a simple UI.  

---

## 🛠️ Tech Stack
- **Python 3.9+**
- [Streamlit](https://streamlit.io/)
- [SQLite3](https://www.sqlite.org/)
- [SentenceTransformers](https://www.sbert.net/)
- [Google Generative AI (Gemini)](https://ai.google/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

## 📂 Project Structure
```
📦 rag-admin-chatbot
 ┣ 📜 app.py              # Main Streamlit app
 ┣ 📜 users.db            # SQLite database (auto-created)
 ┣ 📜 requirements.txt    # Dependencies
 ┣ 📜 .env                # Environment variables (Gemini API Key)
 ┣ 📜 README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/rag-admin-chatbot.git
cd rag-admin-chatbot
```

2️⃣ **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate    # On Windows
```

3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Set Up Environment Variables**  
Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

5️⃣ **Run the App**
```bash
streamlit run main.py
```

---

## 🔑 Default Admin Login
- Email: `admin@example.com`  
- You can change or add more admin emails in the **`admin_emails`** table.

---

## 💡 Example Commands
- `Add the user "john.smith@xyz.com" with phone number "+92332"`  
- `Remove the user "john.smith@xyz.com"`  
- `Update Samantha's city to Cordoba`  
- `Search for users in the Engineering department`  

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
