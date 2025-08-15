from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from langchain.schema import messages_from_dict, messages_to_dict
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_KEY")

app = Flask(__name__, static_folder='static')
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
            
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="gemma2-9b-it"
        )
        
        # pass user id to load_config function to get user settings
        config = load_config(request.args.get('user_id', 'default'))
        template = f"""{{history}}
{config['system_prompt']} The user's name is {config['username']}.
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

@app.route('/save_config', methods=['POST'])
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

@app.route('/get_config')
def get_config():
    user_id = request.args.get('user_id', 'default')
    if (user_id == 'guest'):
        return jsonify({
            "username": "Guest",
            "system_prompt": "You are a digital assistant called Ova. You are here to help me with my tasks. Use new lines for better readability."
        })
    return jsonify(load_config())

@app.route('/get_users')
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
    
    return jsonify(users)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
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
        title_llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="gemma2-9b-it"
        )
        title_prompt = "Generate a very short (2-4 words) title for this conversation based on: " + message
        title = title_llm.predict(title_prompt).strip()
        title = ' '.join(title.split('\n')[0].split()[:4])
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

@app.route('/get_conversations')
def get_conversations():
    chat_history = load_chat_history()
    return jsonify(chat_history)

@app.route('/delete_conversation/<conversation_id>', methods=['DELETE'])
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

@app.route('/get_pfp_type')
def get_pfp_type():
    user_id = request.args.get('user_id', 'default')
    gif_path = os.path.join(app.static_folder, f"pfp_user_{user_id}.gif")
    jpg_path = os.path.join(app.static_folder, f"pfp_user_{user_id}.jpg")
    
    if os.path.exists(gif_path):
        return jsonify({"type": "gif", "path": f"pfp_user_{user_id}.gif"})
    elif os.path.exists(jpg_path):
        return jsonify({"type": "jpg", "path": f"pfp_user_{user_id}.jpg"})
    return jsonify({"type": None, "path": None})

@app.route('/upload_pfp', methods=['POST'])
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

@app.route('/cleanup_memory/<conversation_id>', methods=['POST'])
def cleanup_memory(conversation_id):
    if conversation_id in conversation_memories:
        del conversation_memories[conversation_id]
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True)