# Ova AI thingy? (idk a project name help)

I built ts project quite a while ago and strangly never uploaded to github so nows the time. Even though it was a while ago i remember it taking a longg time to make but in the end turned out great so im glad. Theres quite a few small issues which tbh i forgot what they were.. but one big one that i'll mention is markdown text doesnt get formatted correctly until the chat (or maybe page..) gets refreshed so keep that in mind. **So what is it bruzz?** Pretty much its a container for GroqCloud, i went with this nice minimalistic design with smooth fading animations, thats pretty much it.. read more bru idk.

## Features

- User management with pfps and seperate chat history and configs
- Full memory, AI gets fed memory for previous messages in conversations (memory does not spread across multiple chats that would be hell to implement)
- Customizable system prompt and username in settings
- Minimalistic chat UI with Markdown rendering
- Quick action buttons for common tasks

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/xHeXifx/OvaWeb.git
cd ovaweb
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

```
GROQ_KEY=your_groq_api_key_here
```

### 5. Run the App

```bash
python app.py
```

If you wish to have the app not have a console attached with it change the file extension from app.py to app.pyw

Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Key requirement explanation.
Having a .env file created AND with a value assigned to GROQ_KEY is REQUIRED for Ova to function in any way.

- If .env exists but GROQ_KEY isnt assigned a value: A red error box will appear stating that "GROQ_KEY not found in .env file. Please add your Groq API key."

- If .env doesnt exist: A red error box will appear stating "No .env file found. Please create one with your GROQ_KEY."

- No matter what if one of these errors occur ALL api routes get blocked and return 403. 

---

## Customization

### Change Default Config

- If you want to personalise the code a bit more you can easily edit the following things.

Edit the default config in `app.py` (lines ~47 and ~184):

```python
default_config = {
    "username": "User",
    "system_prompt": "You are a digital assistant called Ova. You are here to help me with my tasks. Use new lines for better readability."
}
```

### Adjust Typing Animation Speed

In `templates/index.html`, search for `typeMessage` function (around line 400):

```javascript
await new Promise(resolve => setTimeout(resolve, 5)); // <-- Increase/decrease for speed
```

Change `5` to a lower value for faster typing, higher for slower.

### Change Welcome Screen Quick Actions

Edit the HTML in `templates/index.html` (around line 600):

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
- Make sure your `.env` file is present and contains your Groq API key.

# Credit
- [My amazing self :3](https://hexif.vercel.app)
- GroqCloud - Hosting the great LLMs i used, tyvm

---

# ⚠️ Known Issues ⚠️
- Users wont be displayed in the login screen if a pfp isnt added alongside it
- When creating a user (with a pfp) they arent logged in, a page refresh is required and then choosing the user otherwise chat elements dont start
- Chat titles often bug out and the model creates incorrect/nonsense titles.

## Developer Documentation

See [how-it-works.md](how-it-works.md) for an developer overview of ts project.