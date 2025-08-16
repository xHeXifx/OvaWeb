# Ova AI thingy? (idk a project name help)

- Ova AI is a container for locally or online hosted AIs. Users are given the option to use a local model served via Ollama or a online model using GroqCloud. 
- The webapp itself allows for a multi-user setup with seperate system prompts and chat histories for each user; This allows you to have seperate users for seperate things (e.g. one for studying and another for code assistance)

## Features

- User management with pfps and seperate chat history and configs
- Full memory, AI gets fed memory for previous messages in conversations (memory does not spread across multiple chats that would be hell to implement)
- Customizable system prompt and username in settings
- Minimalistic chat UI with Markdown rendering
- Quick action buttons for common tasks

## Getting Started

### [GroqCloud Setup (Online)](groq-setup.md)

### [Locally Hosted Setup (Offline)](local-setup.md)

If you wish to have the app not have a console attached with it change the file extension from app.py to app.pyw

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Key requirement explanation.
Having a .env file created AND with a backend variable with the required attachements is REQUIRED for Ova to function in any way.

- If .env exists but a variable isnt assigned a value: A red error box will appear stating that "{Variable} not found in .env file."

- If .env doesnt exist: A red error box will appear stating "No .env file found."

- No matter what if one of these errors occur ALL api routes get blocked and return 403. 

---

## Customization

### Change Default Config

- If you want to personalise the code a bit more you can easily edit the following things.

Edit the default config in `app.py` (lines ~181 and ~443):

```python
default_config = {
    "username": "User",
    "system_prompt": "You are a digital assistant called Ova. You are here to help me with my tasks. Use new lines for better readability."
}
```

### Adjust Typing Animation Speed

In `templates/index.html`, search for `typeMessage` function (around line 1168):

```javascript
await new Promise(resolve => setTimeout(resolve, 5)); // <-- Increase/decrease for speed
```

Change `5` to a lower value for faster typing, higher for slower.

### Change Welcome Screen Quick Actions

Edit the HTML in `templates/index.html` (around line 1435):

```html
<button class="quick-action" onclick="quickAction('Create me a script that ')">
    <i class="fas fa-terminal"></i>
    <span>Create me a script that...</span>
</button>
<!-- ...other quick actions... -->
```
Make sure to edit both quickAction in the button and the span tag

### Modify UI Styles

All CSS is in the `<style>` block at the top of `templates/index.html`. Change colors, fonts, or layout as you wish.

## Troubleshooting

- If you see errors about missing modules, run `pip install -r requirements.txt` again.
- Make sure your `.env` file is present and contains the neccesary variables.
- DM me on discord @hexif 👍

# Credit
- [My amazing self :3](https://hexif.vercel.app)
- GroqCloud - Hosting the great LLMs used, tyvm
- Ollama - Serving locally hosted models

---

# ⚠️ Known Issues ⚠️
- Users wont be displayed in the login screen if a pfp isnt added alongside it
- When creating a user (with a pfp) they arent logged in, a page refresh is required and then choosing the user otherwise chat elements dont start
- Chat titles often bug out and the model creates incorrect/nonsense titles.
- Not really an issue however i am aware the cloud/comptuter icon is off center.. i tried my best to fix it

## Developer Documentation

See [how-it-works.md](how-it-works.md) for an developer overview of ts project.