# Locally Hosted Setup

## 1. Clone the Repository

```bash
git clone https://github.com/xHeXifx/OvaWeb.git
cd OvaWeb
```

## 2. Install Dependencies

Make sure you have Python 3.8+ installed. (I use 3.11.9 if something doesnt work)

```bash
pip install -r requirements.txt
```

## 3. Setup Ollama

### 3.1. Install Ollama
Go to [Ollama's Website](https://ollama.com/) then download and install Ollama for your OS.

### 3.2. Expose Ollama
In Ollama's GUI settings, expose it to your network. This is found under the setting "Expose Ollama to the network".

### 3.3. Pull a Model
- Head to [https://ollama.com/search](https://ollama.com/search) and find the model you want to use.
- Take note of the model name which should be something like "{model}:latest" you'll need this for the config later. Remember these models run off your host so dont choose one that your PC cant handle.
- Run ```ollama pull {modelname}``` this may take a while as the models are quite big

### 3.4. Set Ollama details in ```.env```
Finally set your ollama details in the ```.env``` file of root.
The default OLLAMA_HOST value should be okay however OLLAMA_MODEL will need to be assigned to the model you pulled earlier **including the text after ":"**.

## 4. Set backend type
Again in ```.env``` set the backend variable to "local" This should already be there but commented out so you can remove the comments.


## 5. Run the App

```bash
python app.py
```

### [Back to README](README.md#locally-hosted-setup-offline)