import os
import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)

class FLANT5Trainer:
    """
    A class to train and generate social media posts using the Flan-T5 Small model.

    Attributes:
        model_name (str): Name of the pre-trained model.
        tokenizer (T5Tokenizer): Tokenizer for processing text.
        model (T5ForConditionalGeneration): Fine-tuned Flan-T5 model.
        device (torch.device): Device for computation (MPS for MacBook M1/M2).
    """

    def __init__(self, model_name="google/flan-t5-small"):
        """
        Initializes the Flan-T5 trainer with tokenizer and model.
        Detects MPS (Apple Silicon GPU) if available.
        """
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def load_and_preprocess_data(self, dataset_path):
        """
        Loads and preprocesses JSON dataset.

        Args:
            dataset_path (str): Path to the JSON dataset.

        Returns:
            Dataset: Preprocessed dataset ready for training.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Ensure `image_paths` is always a list
        for entry in raw_data:
            if isinstance(entry.get("image_paths"), str):
                entry["image_paths"] = []

        # Format dataset
        formatted_data = [
            {
                "input_text": f"Generate a social media post with the following details:\n"
                              f"Platform: {entry['platform']}\n"
                              f"Post Heading: {entry['post_heading']}\n"
                              f"Post Content: {entry['post_content']}\n"
                              f"Hashtags: {', '.join(entry['hashtags'])}\n"
                              f"Emojis: {', '.join(entry['emoji'])}",
                "output_text": f"Post Heading: {entry['post_heading']}\n"
                               f"Post Content: {entry['post_content']}\n"
                               f"Hashtags: {', '.join(entry['hashtags'])}\n"
                               f"Emojis: {', '.join(entry['emoji'])}"
            }
            for entry in tqdm(raw_data, desc="Formatting dataset")
        ]

        return Dataset.from_list(formatted_data)

    def tokenize_data(self, dataset):
        """
        Tokenizes dataset for training.

        Args:
            dataset (Dataset): Preprocessed dataset.

        Returns:
            Dataset: Tokenized dataset ready for training.
        """
        def preprocess_function(examples):
            inputs = self.tokenizer(
                examples["input_text"], 
                padding="max_length", 
                truncation=True, 
                max_length=512
            )
            labels = self.tokenizer(
                examples["output_text"], 
                padding="max_length", 
                truncation=True, 
                max_length=128
            ).input_ids

            # Replace padding token ID with -100 for loss computation
            labels = [[label if label != self.tokenizer.pad_token_id else -100 for label in example] for example in labels]
            
            inputs["labels"] = torch.tensor(labels)
            return inputs

        return dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "output_text"])

    def train(self, dataset_path, batch_size=8, epochs=3):
        """
        Trains the Flan-T5 model on the provided dataset.

        Args:
            dataset_path (str): Path to the JSON dataset.
            batch_size (int, optional): Training batch size. Defaults to 8.
            epochs (int, optional): Number of training epochs. Defaults to 3.
        """
        dataset = self.load_and_preprocess_data(dataset_path)
        tokenized_dataset = self.tokenize_data(dataset)

        training_args = TrainingArguments(
            output_dir="./flan_t5_trained",
            evaluation_strategy="no",
            per_device_train_batch_size=batch_size,
            learning_rate=5e-5,
            weight_decay=0.01,
            save_total_limit=2,
            save_strategy="epoch",
            num_train_epochs=epochs,
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
        )

        trainer.train()

        # Save the fine-tuned model
        self.model.save_pretrained("./flan_t5_trained")
        self.tokenizer.save_pretrained("./flan_t5_trained")

    def generate_post(self, topic, temperature=0.9, top_p=0.95, do_sample=True, num_return_sequences=1, repetition_penalty=1.2):
        """
        Generates a social media post based on a given topic.

        Args:
            topic (str): Topic for the social media post.
            temperature (float, optional): Controls randomness. Defaults to 0.9.
            top_p (float, optional): Nucleus sampling. Defaults to 0.95.
            do_sample (bool, optional): Enables sampling. Defaults to True.
            num_return_sequences (int, optional): Number of posts to generate. Defaults to 1.
            repetition_penalty (float, optional): Controls repetition. Defaults to 1.2.

        Returns:
            list: List of generated social media posts.
        """
        prompt = f"Generate a social media post about {topic} with a heading, emojis, and relevant hashtags."
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            max_length=150,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty
        )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in output_ids]

# Usage example
if __name__ == "__main__":
    trainer = FLANT5Trainer()

    # Train the model
    dataset_path = "./data/mixed_curated/mixed_curated.json"
    trainer.train(dataset_path, batch_size=8, epochs=3)

    # Generate social media posts
    topic = "Artificial Intelligence in Marketing"
    generated_posts = trainer.generate_post(topic, temperature=0.7, top_p=0.9, num_return_sequences=3)

    print("### Generated Posts ###")
    for i, post in enumerate(generated_posts, 1):
        print(f"\nPost {i}: {post}")