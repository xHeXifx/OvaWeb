# How Ova Chat Works – Developer Overview

This document explains in detail how ts works ^w^

---

## Backend (`app.py`)

- **Framework:** Flask
- **LLM:** Groq via LangChain (`ChatGroq`)
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

1. On message, loads/creates a `ConversationChain` for the conversation.
2. Uses user's system prompt and username in the prompt template.
3. Stores messages and memory buffer in JSON.
4. For new conversations, generates a short title using the LLM.

---

## Frontend (`templates/index.html`)

- **UI:** Sidebar for conversations, main chat area, modals for settings and deletion
- **User Selection:** Overlay for picking/creating user profiles
- **Chat:** Animated message rendering, Markdown/code support, typing effect
- **Settings:** Modal for editing username, system prompt, and uploading profile picture

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
- To change the LLM model, modify `model_name` in `app.py`.

---

## Security Notes

- No authentication; users are identified by profile selection.
- All data is stored locally; not suitable for production without further security.

---

## File Overview

- `app.py`: Flask backend, LLM integration, user/conversation management
- `templates/index.html`: UI, JavaScript logic, CSS
- `requirements.txt`: Python dependencies

---

## Note

- Admittedly i did get AI to write this markdown file.. as i havent worked on ts for ages i dont fully remember how the flow works and it was just easier to chuck it to an AI instead of reading through the entire flow and writing ts. Forgive me 😭

---