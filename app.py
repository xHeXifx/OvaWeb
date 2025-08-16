from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import json
from datetime import datetime
import os
import requests
from typing import Any, Dict, List, Mapping, Optional
from werkzeug.utils import secure_filename
from langchain.schema import messages_from_dict, messages_to_dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_KEY")
groq_model = os.getenv("GROQ_MODEL")

backend = os.getenv("backend")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

class OllamaLLM(LLM):
    
    host: str = None
    model: str = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, host: str, model: str):
        super().__init__()
        self.host = host
        self.model = model
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"host": self.host, "model": self.model}
        
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """call the Ollama API with the given prompt"""
        try:
            url = f"{self.host}/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                error_msg = f"Ollama API returned error: {response.status_code} - {response.text}"
                raise ValueError(error_msg)
                
            result = response.json()
            return result.get("response", "")
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}. Make sure Ollama is running at {self.host}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response from Ollama API: {response.text}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error when calling Ollama: {str(e)}")

app = Flask(__name__, static_folder='static')

@app.before_request
def check_env_variables():
    if request.endpoint == 'static':
        return
    
    env_error = None
    if not os.path.exists('.env'):
        env_error = "No .env file found. Please create one with your configuration."
    elif backend == "groq" and not groq_key:
        env_error = "GROQ_KEY not found in .env file. Please add your Groq API key when using 'groq' backend."
    elif backend == "local" and (not OLLAMA_HOST or not OLLAMA_MODEL):
        if not OLLAMA_HOST:
            env_error = "OLLAMA_HOST not found in .env file. Please add your Ollama host URL when using 'local' backend."
        elif not OLLAMA_MODEL:
            env_error = "OLLAMA_MODEL not found in .env file. Please specify which Ollama model to use."
    elif backend not in ["groq", "local"]:
        env_error = "Invalid backend specified in .env file. Please use 'groq' or 'local'."
    
    if env_error:
        if request.path.startswith('/api/'):
            return jsonify({"error": "env_error", "message": env_error}), 403
        return render_template('index.html', env_error=env_error)
os.makedirs('static', exist_ok=True)
CHAT_HISTORY_FILE = "chat_history.json"

# memory for active conversations
conversation_memories = {}

def get_conversation_chain(conversation_id):
    if (conversation_id not in conversation_memories):
        # make new conversation with memory
        memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Assistant")
        
        # load existing memory if it exists
        chat_history = load_chat_history()
        if conversation_id in chat_history and 'memory_buffer' in chat_history[conversation_id]:
            memory_messages = messages_from_dict(chat_history[conversation_id]['memory_buffer'])
            memory.chat_memory.messages = memory_messages
        
        # init LLM based on backend setting
        if backend == "groq":
            llm = ChatGroq(
                groq_api_key=groq_key,
                model_name=groq_model
            )
        elif backend == "local":
            try:
                llm = OllamaLLM(
                    host=OLLAMA_HOST,
                    model=OLLAMA_MODEL
                )
            except Exception as e:
                error_msg = f"Failed to initialize Ollama: {str(e)}"
                raise RuntimeError(error_msg)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # pass user id to load_config function to get user settings
        config = load_config(request.args.get('user_id', 'default'))
        
        template = f"""{config['system_prompt']} The user's name is {config['username']}.

{{history}}
Human: {{input}}
Assistant: """
        
        prompt = PromptTemplate(
            input_variables=["history", "input"], 
            template=template
        )
        
        conversation_memories[conversation_id] = ConversationChain(
            llm=llm,
            verbose=True,
            memory=memory,
            prompt=prompt
        )
    
    return conversation_memories[conversation_id]

def load_chat_history():
    user_id = request.args.get('user_id')
    chat_history_file = f"chat_history_{user_id}.json"
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r') as f:
            return json.load(f)
    return {}

def save_chat_history(history):
    user_id = request.args.get('user_id')
    chat_history_file = f"chat_history_{user_id}.json"
    with open(chat_history_file, 'w') as f:
        json.dump(history, f, indent=4)

def load_config(user_id=None):
    if user_id is None:
        user_id = request.args.get('user_id', 'default')
    config_file = f"config_user_{user_id}.json"
    default_config = {
        "username": "User",
        "system_prompt": "You are a digital assistant called Ova. You are here to help me with my tasks. Use new lines for better readability."
    }
    
    # try to load users config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return default_config

@app.route('/api/get_backend_info')
def get_backend_info():
    if backend == "groq":
        return jsonify({
            "type": "cloud",
            "model": groq_model
        })
    elif backend == "local":
        return jsonify({
            "type": "local",
            "model": OLLAMA_MODEL
        })
    else:
        return jsonify({
            "type": "unknown",
            "model": "unknown"
        })

@app.route('/api/save_config', methods=['POST'])
def save_config():
    user_id = request.args.get('user_id', 'default')
    config = request.json
    
    # handle new user creation
    if user_id == 'new':
        user_id = datetime.now().strftime('%Y%m%d%H%M%S')
    
    config_file = f"config_user_{user_id}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    return jsonify({"success": True, "user_id": user_id})

@app.route('/api/get_config')
def get_config():
    user_id = request.args.get('user_id', 'default')
    if (user_id == 'guest'):
        config = {
            "username": "Guest",
            "system_prompt": "You are a digital assistant called Ova. You are here to help me with my tasks. Use new lines for better readability."
        }
    else:
        config = load_config()
    
    config['backend_type'] = backend
    if backend == "local" and OLLAMA_MODEL:
        config['model_name'] = OLLAMA_MODEL
    elif backend == "groq":
        config['model_name'] = groq_model
    
    return jsonify(config)

@app.route('/api/get_users')
def get_users():
    users = []
    
    for filename in os.listdir(app.static_folder):
        if filename.startswith('pfp_') and (filename.endswith('.jpg') or filename.endswith('.gif')):
            # get user_id from filename
            user_id = filename.split('_')[2].split('.')[0]
            config_file = f"config_user_{user_id}.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    users.append({
                        'id': user_id,
                        'username': config.get('username', 'User'),
                        'image': filename
                    })

    # add guest option
    users.append({
        'id': 'guest',
        'username': 'Guest',
        'image': None
    })
    
    return jsonify({'users': users})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    conversation_id = data.get('conversation_id')
    is_new_conversation = data.get('is_new_conversation', False)
    
    if is_new_conversation:
        conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        # clean up memory if it's a new conversation
        if conversation_id in conversation_memories:
            del conversation_memories[conversation_id]

    chat_history = load_chat_history()
    
    if conversation_id not in chat_history:
        chat_history[conversation_id] = {'messages': [], 'title': None}

    conversation = get_conversation_chain(conversation_id)
    
    response = conversation.predict(input=message)

    # for new conversations generate title
    if is_new_conversation:
        title_prompt = "Create a very short title (2-4 words only) for a conversation that starts with this message. ONLY respond with the title, no other text: " + message
        
        # use the chosen backend for title generation
        if backend == "groq":
            title_llm = ChatGroq(
                groq_api_key=groq_key,
                model_name=groq_model
            )
            title_response = title_llm.predict(title_prompt).strip()
        elif backend == "local":
            try:
                title_llm = OllamaLLM(
                    host=OLLAMA_HOST,
                    model=OLLAMA_MODEL
                )
                title_response = title_llm._call(title_prompt).strip()
            except Exception as e:
                # fallback to a generic title if Ollama fails
                app.logger.error(f"Failed to generate title with Ollama: {str(e)}")
                title_response = f"Conversation {conversation_id[:6]}"
        else:
            title_response = f"Conversation {conversation_id[:6]}"
        
        # clean up the title - the ai was being annoying and kept adding prefix words before the title so this is here just in case
        common_prefixes = ["sure", "here", "is", "a", "the", "title", "for", "this", "conversation", "would", "be", "could", "i", "think", "based", "on"]
        
        # filter out common prefixes
        words = title_response.lower().split()
        start_idx = 0
        
        # find where the actual title
        for i, word in enumerate(words):
            if word.strip(',.!?:;') not in common_prefixes:
                start_idx = i
                break
        
        title_words = title_response.split()[start_idx:start_idx+4]
        
        if not title_words and len(words) > 0:
            title_words = title_response.split()[-4:]
        
        # join words and capitalize
        title = ' '.join(title_words).strip()
        title = ' '.join(word.capitalize() for word in title.split())
        
        # use fallback if titles still somehow empty
        if not title:
            title = f"Conversation {conversation_id[:6]}"
            
        chat_history[conversation_id]['title'] = title

    # update chat history ...
    chat_history[conversation_id]['messages'].append({
        "role": "user",
        "content": message
    })
    chat_history[conversation_id]['messages'].append({
        "role": "assistant",
        "content": response
    })
    
    if conversation_id in conversation_memories:
        memory_dict = messages_to_dict(conversation_memories[conversation_id].memory.chat_memory.messages)
        chat_history[conversation_id]['memory_buffer'] = memory_dict
    
    # ... then call save function here vv ^o^
    save_chat_history(chat_history)
    
    return jsonify({
        "response": response,
        "conversation_id": conversation_id,
        "title": chat_history[conversation_id]['title']
    })

@app.route('/api/get_conversations')
def get_conversations():
    chat_history = load_chat_history()
    return jsonify(chat_history)

@app.route('/api/delete_conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"success": False, "error": "No user ID provided"}), 400
        
    chat_history = load_chat_history()
    if conversation_id in chat_history:
        if conversation_id in conversation_memories:
            del conversation_memories[conversation_id]
        del chat_history[conversation_id]
        save_chat_history(chat_history)
        return jsonify({"success": True})
    return jsonify({"success": False}), 404

@app.route('/api/get_pfp_type')
def get_pfp_type():
    user_id = request.args.get('user_id', 'default')
    gif_path = os.path.join(app.static_folder, f"pfp_user_{user_id}.gif")
    jpg_path = os.path.join(app.static_folder, f"pfp_user_{user_id}.jpg")
    
    if os.path.exists(gif_path):
        return jsonify({"type": "gif", "path": f"pfp_user_{user_id}.gif"})
    elif os.path.exists(jpg_path):
        return jsonify({"type": "jpg", "path": f"pfp_user_{user_id}.jpg"})
    return jsonify({"type": None, "path": None})

@app.route('/api/upload_pfp', methods=['POST'])
def upload_pfp():
    if 'pfp' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    user_id = request.args.get('user_id', 'default')
    file = request.files['pfp']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif']:
            return jsonify({"error": "Invalid file type"}), 400
            
        # delete existing profile pictures if they exist
        for existing in [f'pfp_user_{user_id}.jpg', f'pfp_user_{user_id}.gif']:
            try:
                os.remove(os.path.join(app.static_folder, existing))
            except OSError:
                pass
            
        filename = f"pfp_user_{user_id}.gif" if ext == '.gif' else f"pfp_user_{user_id}.jpg"
        file.save(os.path.join(app.static_folder, filename))
        
        # make default config with system prompt for new users
        config_file = f"config_user_{user_id}.json"
        if not os.path.exists(config_file):
            default_config = {
                "username": "New User",
                "system_prompt": "You are a digital assistant called Ova. You are here to help me with my tasks. Use new lines for better readability."
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
                
        return jsonify({"success": True, "filename": filename})

@app.route('/api/cleanup_memory/<conversation_id>', methods=['POST'])
def cleanup_memory(conversation_id):
    if conversation_id in conversation_memories:
        del conversation_memories[conversation_id]
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True)