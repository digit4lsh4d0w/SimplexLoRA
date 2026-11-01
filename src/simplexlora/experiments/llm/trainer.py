from collections.abc import Callable
from typing import Protocol, cast

import torch
import wandb
import peft
import datasets
from peft import PeftMixedModel, PeftModel
from peft.tuners.lora.layer import LoraLayer

from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    pipeline,
)

from simplexlora import optimizers, utils
from simplexlora.experiments.arguments import as_tokenizer, as_trainer_optimizers

from .data import DatasetBuilder, DatasetRegistry

import warnings

warnings.filterwarnings("ignore")

DATASETS = [
    "aqua",
    "gsm8k",
    "commonsensqa",
    "boolq",
    "addsub",
    "multiarith",
    "singleeq",
    "strategyqa",
    "svamp",
    "bigbench_date",
    "object_tracking",
    "coin_flip",
    "last_letters",
    "mathqa",
    "hella_swag",
    "arc_challenge",
]

ModelType = PreTrainedModel | PeftModel | PeftMixedModel


class TextGenerator(Protocol):
    def __call__(self, text: str, **kwargs: object) -> list[dict[str, str]]: ...


class Finetuner:
    """Main class for downstream finetuning"""

    def __init__(self, args):
        self.args = args
        self.model: ModelType | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.builder: DatasetBuilder | None = None
        self.train_dataset: datasets.Dataset | None = None
        self.eval_dataset: datasets.Dataset | None = None
        self.trainer: Trainer | None = None
        self.setup_logging()
        if self.args.wandb:
            self.setup_wandb()

    def setup_logging(self):
        """Setup logging configuration"""
        logger.info(f"Starting downstream finetuning for dataset: {self.args.dataset}")

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        wandb.init(
            project=self.args.wandb_project,
            tags=[self.args.model, self.args.dataset, self.args.optimizer],
            name=self.args.run_name,
            config=self.args,
        )

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configurations"""
        logger.info("Loading model and tokenizer...")

        # Determine optimal settings based on GPU capability
        # [TODO] add flash_attention
        if torch.cuda.get_device_capability()[0] >= 8:
            attn_implementation = "eager"  # fix flash_attention_2
        else:
            attn_implementation = "eager"

        # Set dtype
        if self.args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.args.dtype == "float16":
            torch_dtype = torch.float16
        elif self.args.dtype == "float32":
            torch_dtype = torch.float32
        elif self.args.dtype == "float64":
            torch_dtype = torch.float64
        # Setup quantization config
        if self.args.quant_bit == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
        elif self.args.quant_bit == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
        )

        # Load tokenizer
        self.tokenizer = as_tokenizer(
            AutoTokenizer.from_pretrained(
                self.args.model,
                use_fast=self.args.use_fast_tokenizer,
                padding_side=self.args.padding_side,
            )
        )
        self.tokenizer.pad_token_id = 0

        # Resize token embeddings if necessary
        embeddings = cast(torch.nn.Embedding, self.model.get_input_embeddings())
        embedding_size = embeddings.weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Model and tokenizer loaded successfully.")

    def setup_peft(self):
        """Setup PEFT (Parameter Efficient Fine-Tuning) adapters"""
        logger.info("Setting up PEFT adapters...")

        model = self.model
        if model is None:
            raise RuntimeError("Model must be loaded before configuring PEFT")

        peft_args = utils.get_peft_arguments(self.args)
        if peft_args is not None:
            peft_args.task_type = "CAUSAL_LM"
            self.model = peft.get_peft_model(cast(PreTrainedModel, model), peft_args)

        # Print trainable parameters info
        tr_param_count, all_param_count, tr_persent = utils.print_trainable_params(self.model, verbose=True)

        num_peft_adapters = utils.count_atapters(self.model, self.args.ft_strategy)

        if self.args.wandb:
            wandb.log(
                {
                    "trainable_params_count": tr_param_count,
                    "total_param_count": all_param_count,
                    "trainable_params_percentage": tr_persent,
                    "num_peft_adapters": num_peft_adapters,
                }
            )

    def load_datasets(self):
        """Load and prepare both training and evaluation datasets"""
        logger.info("Loading datasets...")

        # Set dataset paths and create builder
        DatasetRegistry.set_dataset_paths(self.args)
        self.builder = DatasetRegistry.create_builder(self.args)
        if self.builder is None:
            raise RuntimeError("Dataset builder was not created")
        data = self.builder.get_data()

        # Extract train and eval data
        train_questions, train_answers = data["train"]
        eval_questions, eval_answers = data["eval"]

        # Create datasets
        self.train_dataset = None
        self.eval_dataset = None

        if train_questions:
            train_dict = {
                "question": ["Question: " + q for q in train_questions],
                "response": train_answers,
                "raw_x": train_questions,
                "raw_y": train_answers,
            }
            self.train_dataset = datasets.Dataset.from_dict(train_dict)

        if eval_questions:
            eval_dict = {
                "question": ["Question: " + q for q in eval_questions],
                "response": eval_answers,
                "raw_x": eval_questions,
                "raw_y": eval_answers,
            }
            self.eval_dataset = datasets.Dataset.from_dict(eval_dict)

    def prepare_training_dataset(self):
        """Prepare dataset for training"""
        if self.train_dataset is None:
            logger.warning("No training dataset available")
            return None
        if self.builder is None or self.tokenizer is None:
            raise RuntimeError("Dataset builder and tokenizer must be initialized")

        logger.info("Preparing training dataset...")
        return self.builder.preprocess_dataset(
            self.tokenizer,
            self.args.max_seq_length,
            self.args.seed,
            self.train_dataset,
            eval_mode=False,
        )

    def prepare_evaluation_dataset_for_training(self):
        """Prepare evaluation dataset for training"""
        if self.eval_dataset is None:
            logger.warning("No eval dataset for training dataset available")
            return None
        if self.builder is None or self.tokenizer is None:
            raise RuntimeError("Dataset builder and tokenizer must be initialized")

        logger.info("Preparing evaluation dataset for training...")
        return self.builder.preprocess_dataset(
            self.tokenizer,
            self.args.max_seq_length,
            self.args.seed,
            self.eval_dataset,
            eval_mode=False,
        )

    def prepare_evaluation_dataset(self):
        """Prepare dataset for evaluation"""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset available")
            return None
        if self.builder is None:
            raise RuntimeError("Dataset builder must be initialized")

        logger.info("Preparing evaluation dataset...")
        builder = self.builder
        dataset = self.eval_dataset.map(lambda sample: builder.create_prompt_formats(sample, eval_mode=True))
        return dataset

    def train(self):
        """Execute training process"""
        if self.args.do_not_train:
            logger.info("Training skipped (do_not_train=True)")
            return

        logger.info(f"Starting training for dataset: {self.args.dataset}")

        # Setup trainer
        training_args = TrainingArguments(
            do_train=not self.args.do_not_train,
            do_eval=not self.args.do_not_eval,
            do_predict=self.args.do_predict,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=(
                self.args.eval_batch_size if self.args.eval_batch_size else self.args.batch_size
            ),
            gradient_accumulation_steps=self.args.grad_acc_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=(
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else round(self.args.warmup_ratio * self.args.max_steps_train)
            ),
            learning_rate=self.args.lr,
            weight_decay=self.args.wd,
            num_train_epochs=self.args.n_epoches_train,
            max_steps=self.args.max_steps_train,
            logging_steps=self.args.logging_steps,
            eval_strategy=self.args.eval_strategy,
            save_strategy=self.args.eval_strategy,
            eval_steps=self.args.eval_steps,
            save_steps=self.args.eval_steps,
            bf16=(self.args.dtype == "bfloat16"),
            fp16=(self.args.dtype == "float16"),
            output_dir=f"./src/fine_tuning/llm/{self.args.results_path}/{self.args.run_name}",
            run_name=self.args.run_name,
            report_to=["wandb"] if self.args.wandb else ["none"],
            load_best_model_at_end=self.args.eval_strategy != "no",
            save_only_model=True,
            metric_for_best_model=self.args.metric_for_best_model,
            greater_is_better=self.args.metric_for_best_model not in ["loss"],
        )

        model = self.model
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before training")

        # Prepare datasets
        train_dataset = self.prepare_training_dataset()
        if train_dataset is None:
            logger.error(
                "No training data available, so either skip training (--do_not_train) or provide training data"
            )
            return

        eval_dataset = None
        if not self.args.do_not_eval:
            eval_dataset = self.prepare_evaluation_dataset_for_training()
            if eval_dataset is None:
                logger.error(
                    "No evaluation data available, so either skip evaluation (--do_not_eval) or provide evaluation data"
                )
                return

        if self.args.ft_strategy == "SimplexLoRA":
            weight_params = []
            loraAB_params = []
            lora_layers: list[LoraLayer] = []
            other_params = []

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    lora_layers.append(module)

            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    loraAB_params.append(param)
                if "lora_weight" in name:
                    weight_params.append(param)

            optimizer = optimizers.SimplexAdamW(
                [
                    {"params": loraAB_params, "name": "loraAB"},
                    {"params": other_params, "name": "other_params"},
                    {
                        "params": weight_params,
                        "proj": optimizers.proj_simplex_euclidean,
                        "lr": self.args.lr_w,
                        "wd": self.args.wd_w,
                        "name": "weight_params",
                    },
                ],
                lora_layers=lora_layers,
                lr=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                num_adapters=len(weight_params),
                fat_step=self.args.fat_step,
                max_fat_steps=self.args.mfs,
                lora_extention=self.args.extention,
                default_lora_rank=self.args.lora_r,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay
            )
        self.trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            optimizers=as_trainer_optimizers(optimizer, None),
        )

        # Clean up memory before training
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Train the model
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("Trainer was not initialized")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info(f"Training completed. Metrics: {metrics}")

    def evaluate(self):
        """Execute evaluation process"""
        if self.args.do_not_eval:
            logger.info("Evaluation skipped (do_not_eval=True)")
            return 0, 0

        logger.info(f"Starting evaluation for dataset: {self.args.dataset}")

        eval_dataset = self.prepare_evaluation_dataset()

        # Prepare evaluation dataset
        if eval_dataset is None:
            logger.error("No evaluation data available")
            return 0, 0

        model = self.model
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before evaluation")

        # Setup text generation pipeline
        pipeline_factory = cast(Callable[..., TextGenerator], pipeline)
        generator = pipeline_factory(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
        )

        # Evaluate model
        correct, total = 0, 0

        logger.info(f"Evaluation sample prompt: {eval_dataset['text'][0]}")

        for i, item in enumerate(eval_dataset):
            try:
                # Generate prediction
                predicted_response = generator(
                    item["text"],
                    max_new_tokens=len(tokenizer.tokenize(item["raw_y"])) + 1,
                    num_return_sequences=1,
                )[0]["generated_text"]
                predicted_response = predicted_response.replace(" ", "").replace("\n", "")
                item["raw_y"] = item["raw_y"].replace(" ", "").replace("\n", "")
                # Check if correct answer is in prediction
                if item["raw_y"] in predicted_response:
                    correct += 1

                # Debug output
                print(f">>Prediction<<: {predicted_response}")
                print(f">>>>Answer<<<<: {item['raw_y']}")

                total += 1

                # Progress update
                accuracy = (correct / total) * 100 if total > 0 else 0
                print(f"[{i + 1}/{len(eval_dataset)}] Accuracy: {accuracy:.2f}%")
                print("=" * 50)

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue

        return correct, total

    def log_final_results(self, correct, total):
        """Log final evaluation results"""
        if total > 0:
            final_accuracy = (correct / total) * 100
            logger.info(f"[FINAL] Accuracy: {final_accuracy:.2f}%")
            if self.trainer is not None:
                self.trainer.save_metrics("eval", {"accuracy": final_accuracy})
            if self.args.wandb:
                wandb.log({"final_accuracy": final_accuracy})
        else:
            logger.info("No samples were successfully evaluated.")

    def run(self):
        """Main execution flow"""
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load model and setup PEFT
        self.load_model_and_tokenizer()
        self.setup_peft()

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train()
        correct, total = self.evaluate()

        # Log results
        self.log_final_results(correct, total)
        logger.info("Pipeline completed successfully")


def main(args):
    """Main entry point"""
    # Create and run finetuner
    finetuner = Finetuner(args)
    finetuner.run()


if __name__ == "__main__":
    main(None)
