# GroqCloud Setup

### 1. Clone the Repository

```bash
git clone https://github.com/xHeXifx/OvaWeb.git
cd OvaWeb
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed. (I use 3.11.9 if something doesnt work)

```bash
pip install -r requirements.txt
```

### 3. Get a Groq API Key

- Go to [Groq Cloud](https://console.groq.com/) and sign up.
- Navigate to API Keys and create a new key. (https://console.groq.com/keys)
- Copy your API key.

### 4. Set Up Environment Variables

Create a `.env` file in the project root:
Set the following values:

```
backend="groq"
GROQ_KEY="your_groq_api_key_here"
GROQ_MODEL="gemma2-9b-it"
```
You can change the model to any other model provided at [https://console.groq.com/docs/models](https://console.groq.com/docs/models)



### 5. Run the App

```bash
python app.py
```

### [Back to README](README.md#groqcloud-setup-online)