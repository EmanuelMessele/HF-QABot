from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def main():

    context = "Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, etc. in 100+ languages."
    question = "What does Transformers provide?"

    result = qa(question=question, context=context)
    print(f"Answer: {result['answer']}")
    print(f"Confidence Score: {result['score']}")


if __name__ == "__main__":
    main()

