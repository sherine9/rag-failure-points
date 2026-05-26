from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Generator:
    """
    Answer generator using Mistral-7B-Instruct.
    
    Takes a question + retrieved chunks and generates an answer.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Args:
            model_name: HuggingFace model for generation.
        """
        print(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,   
            device_map="auto"      
        )
        self.model.eval()
        print("Generator loaded!")

    def build_prompt(self, question, chunks):
        context = "\n\n".join([f"Passage {i+1}: {c}" for i, c in enumerate(chunks)])
        
        prompt = f"""[INST] You are a precise question answering system.
    Answer the question using ONLY the provided context.
    Give a SHORT answer — a name, date, number, or brief phrase.
    Do NOT write full sentences. Do NOT explain.
    If the answer is not in the context, write "unknown".

    Context:
    {context}

    Question: {question}

    Short answer: [/INST]"""
        return prompt

    def generate(self, question, chunks, max_new_tokens=100):
        prompt = self.build_prompt(question, chunks)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        answer = self._extract_short_answer(answer)
        return answer.strip()

    def _extract_short_answer(self, text):
        """
        Extract a short answer from verbose generation.
        Tries to match the style of NQ Open gold answers.
        """
        text = text.strip()
        
        # remove "I don't know" responses
        if "i don't know" in text.lower() or "not mentioned" in text.lower():
            return text
        
        # take first sentence only
        for sep in [". ", ".\n", ", which", ", but", " (", " - "]:
            if sep in text:
                text = text.split(sep)[0]
        
        # remove common prefixes the model adds
        prefixes = [
            "the answer is ",
            "it was ",
            "it is ",
            "they were ",
            "he was ",
            "she was ",
            "this was ",
            "that was ",
        ]
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):]
                break
        
        return text.strip()


if __name__ == "__main__":
    # quick test
    generator = Generator()
    
    question = "When was the Eiffel Tower built?"
    chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889.",
        "The tower was designed by engineer Gustave Eiffel and built for the 1889 World's Fair."
    ]
    
    answer = generator.generate(question, chunks)
    print(f"Question: {question}")
    print(f"Answer: {answer}")