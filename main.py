from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import torch

model_path = "model/gemma-3-ib-it"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if torch.cuda.is_available():
    model = model.to("cuda")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("structure.html")

@app.route("/ask", methods=["POST"])

def ask():
    data = request.get_json()
    user_text = data.get("text", "")

    message = [{"role" : "user", "content" : user_text}]
    
    inputs = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens = True)
    
    def run_model():
        model.generate(**inputs, streamer=streamer, max_new_tokens=30, eos_token_id=tokenizer.eos_token_id)

    threading.Thread(target=run_model).start()

    def generate():
        for token in streamer:
            yield token

            if tokenizer.eos_token in token:
                break

    return Response(stream_with_context(generate()), mimetype="text/plain")
if __name__ == "__main__":
    app.run()