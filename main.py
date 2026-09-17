import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice


# Load directly from Hugging Face Hub
MODEL_PATH = "Nihal23f2000003/smart-mcq-roberta"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForMultipleChoice.from_pretrained(
    MODEL_PATH
)

model.eval()


def predict(
    question,
    option_a,
    option_b,
    option_c,
    option_d,
    option_e
):

    options = [
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]

    # Same preprocessing as training
    first_sentences = [question] * 5
    second_sentences = options

    inputs = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    # Shape: [1, 5, sequence_length]
    inputs = {
        key: value.unsqueeze(0)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    labels = ["A", "B", "C", "D", "E"]

    confidence = {
        labels[i]: float(probabilities[i])
        for i in range(5)
    }

    predicted_index = torch.argmax(probabilities).item()
    predicted_answer = labels[predicted_index]

    top3_indices = torch.argsort(
        probabilities,
        descending=True
    )[:3]

    top3_answers = " → ".join(
        labels[i.item()]
        for i in top3_indices
    )

    return predicted_answer, top3_answers, confidence


demo = gr.Interface(
    fn=predict,

    inputs=[
        gr.Textbox(label="Question", lines=3),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E")
    ],

    outputs=[
        gr.Textbox(label="🏆 Predicted Answer"),
        gr.Textbox(label="🥇 Top 3 Ranking"),
        gr.Label(label="📊 Confidence Scores")
    ],

    title="🧠 Smart MCQ Solver",

    description="""
    AI-powered Multiple Choice Question Solver using a
    fine-tuned RoBERTa model.
    Enter a question and five options to receive predictions.
    """
)


if __name__ == "__main__":
    demo.launch()   