from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn

app = Flask(__name__)

class DistilBertClassifier(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output["last_hidden_state"]
        cls_hidden_state = hidden_state[:, 0, :]
        cls_hidden_state = self.drop(cls_hidden_state)
        output = self.linear(cls_hidden_state)
        return output

model_config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
model = DistilBertClassifier(bert)
model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def classify_text(text, model, tokenizer):
    tokenized_text = tokenizer(text, return_tensors="pt")
    tokenized_text = {k: v for k, v in tokenized_text.items()}
    with torch.no_grad():
        model_output = model(**tokenized_text)
    pred = torch.sigmoid(model_output).item()
    return {"prediction": "positive" if pred > 0.5 else "negative", "score": pred}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    text = data.get('text', '')
    result = classify_text(text, model, tokenizer)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
