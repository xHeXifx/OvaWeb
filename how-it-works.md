# How Ova Chat Works – Developer Overview

This document explains in detail how ts works ^w^

---

## Backend (`app.py`)

- **Framework:** Flask
- **LLM Options:**
  - **Cloud:** Groq via LangChain (`ChatGroq`) using model specified in `.env`
  - **Local:** Ollama via custom `OllamaLLM` class using any model available in your Ollama installation
- **Backend Selection:** Controlled via environment variables in `.env` file
- **Memory:** Per-conversation memory using `ConversationBufferMemory`
- **Persistence:** User configs and chat histories stored as JSON files on disk
- **Profile Pictures:** Uploaded and stored in `/static` as `pfp_user_{user_id}.jpg` or `.gif`

### Key Endpoints

- `/chat`: Receives user message, generates response, updates history
- `/get_conversations`: Returns all conversations for a user
- `/save_config` & `/get_config`: Manage user settings (username, system prompt)
- `/get_users`: Lists all users with profile pictures
- `/upload_pfp`: Handles profile picture upload
- `/delete_conversation/<id>`: Deletes a conversation
- `/cleanup_memory/<id>`: Removes in-memory conversation state

### Conversation Flow

1. On message, loads/creates a `ConversationChain` for the conversation with the appropriate backend (Groq or Ollama).
2. Uses user's system prompt and username in the prompt template.
3. Stores messages and memory buffer in JSON.
4. For new conversations, generates a short title using the selected LLM.

---

## Frontend (`templates/index.html`)

- **UI:** Sidebar for conversations, main chat area, modals for settings and deletion
- **User Selection:** Overlay for picking/creating user profiles
- **Chat:** Animated message rendering, Markdown/code support, typing effect
- **Settings:** Modal for editing username, system prompt, and uploading profile picture
- **Backend Indicator:** Visual indicator showing whether using cloud (Groq) or local (Ollama) backend with model name on hover

### JavaScript Structure

- **User Management:** Loads users, handles selection, supports guest/new user
- **Conversation Management:** Loads/saves conversations, deletes, starts new chat
- **Message Rendering:** `createMessage` and `typeMessage` functions handle animated display and Markdown parsing
- **Settings Modal:** Loads/saves config, handles profile picture preview/upload
- **Quick Actions:** Buttons on welcome screen pre-fill input for common tasks

### Customization Points

- **Typing Speed:** `typeMessage` function, change delay in `setTimeout`
- **Default Prompts/Usernames:** Change in backend (`app.py`) or frontend modal defaults
- **UI Styles:** All CSS in `<style>` block at top of HTML

---

## Data Storage

- **User Config:** `config_user_{user_id}.json`
- **Chat History:** `chat_history_{user_id}.json`
- **Profile Pictures:** `/static/pfp_user_{user_id}.jpg` or `.gif`

---

## Adding Features

- To add new quick actions, edit the welcome screen HTML.
- To support more file types for profile pictures, update the allowed extensions in `app.py` (`upload_pfp` route).
- To change the LLM model:
  - For Groq: Modify the GROQ_MODEL variable in your .env file

  - For Ollama: Change the OLLAMA_MODEL in your .env file

---

## Security Notes

- SHA256 used for user passcodes however can be easily changed to something else by changing json
- All data is stored locally; not suitable for production without further security.
- When using local backend (Ollama), no API keys are sent to external services.
- When using cloud backend (Groq), API key is required and stored in the .env file.

---

## File Overview

- `app.py`: Flask backend, LLM integration, user/conversation management
- `templates/index.html`: UI, JavaScript logic, CSS
- `requirements.txt`: Python dependencies
- `.env`: Configuration file for backend selection and API keys
- `groq-setup.md`: Instructions for setting up Groq cloud backend
- `local-setup.md`: Instructions for setting up Ollama local backend
- `updater.py`: Automatic script updater for OvaWeb. Fetches new version and copies new files over automatically.

---