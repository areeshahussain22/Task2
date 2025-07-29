from datasets import load_dataset
import gradio as gr
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

dataset = load_dataset("ag_news")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
dataset_tokenized = dataset.map(tokenize, batched=True)
dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
training_args = TrainingArguments(
    output_dir="bert_ag_news",
    evaluation_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_steps=100,
    save_total_limit=1,
    logging_dir="logs"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    compute_metrics=compute_metrics
)
trainer.train()
metrics = trainer.evaluate()
print(metrics)
model.save_pretrained("bert_ag_news")
tokenizer.save_pretrained("bert_ag_news")
from transformers import BertTokenizerFast, BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert_ag_news")
tokenizer = BertTokenizerFast.from_pretrained("bert_ag_news")
model.eval()

def predict_topic(headline: str):
    inputs = tokenizer(headline, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1).squeeze().tolist()
    topics = [ "Sci/Tech","Sports", "Business"]
    return {topics[i]: probs[i] for i in range(4)}

iface = gr.Interface(
    fn=predict_topic,
    inputs=gr.Textbox(label="News Headline"),
    outputs=gr.Label(num_top_classes=4, label="Predicted Topic"),
    title=" News Classifier",
    description="Enter a news headline to get a predicted topic."
)

iface.launch(share=True)
