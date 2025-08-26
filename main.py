from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "model/gemma-3-ib-it"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

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

    outputs = model.generate(**inputs, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)

    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

    return jsonify({"reply":reply})

if __name__ == "__main__":
    app.run()