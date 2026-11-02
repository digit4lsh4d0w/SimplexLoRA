"""Data-contract tests for Decompile-Bench prompts, labels and collation."""

from __future__ import annotations

import unittest
from typing import cast

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from simplexlora.experiments.decompile import CausalLMCollator, build_messages, tokenize_dataset


class _CharacterTokenizer:
    pad_token_id = 0

    @staticmethod
    def _render(messages: list[dict[str, str]]) -> str:
        return "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool = False,
    ) -> str:
        if tokenize:
            raise AssertionError("test tokenizer only supports text chat templates")
        rendered = self._render(messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if add_special_tokens:
            raise AssertionError("decompile data should not add implicit special tokens")
        return {"input_ids": [ord(character) for character in text]}


class DecompileDataContractTests(unittest.TestCase):
    def test_messages_use_assembly_as_input_and_source_as_assistant_output(self) -> None:
        row = {
            "name": "func0",
            "asm": "mov $1, %eax\nretq",
            "code": "int func0(void) { return 1; }",
        }
        messages = build_messages(row)

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertIn("func0:", messages[1]["content"])
        self.assertEqual(messages[1]["content"].count(row["asm"]), 1)
        self.assertEqual(messages[2]["content"], row["code"])

    def test_prompt_tokens_are_masked_and_completion_tokens_are_targets(self) -> None:
        tokenizer = _CharacterTokenizer()
        dataset = Dataset.from_dict(
            {
                "name": ["func0"],
                "asm": ["mov $1, %eax\nretq"],
                "code": ["int func0(void) { return 1; }"],
                "file": ["project/src/func0.c"],
            }
        )

        tokenized = tokenize_dataset(
            dataset,
            cast(PreTrainedTokenizerBase, tokenizer),
            max_seq_length=10_000,
        )
        row = tokenized[0]
        prompt_text = tokenizer.apply_chat_template(
            build_messages(dataset[0])[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_length = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        self.assertEqual(row["labels"][:prompt_length], [-100] * prompt_length)
        self.assertEqual(row["labels"][prompt_length:], row["input_ids"][prompt_length:])
        self.assertGreater(len(row["labels"]) - prompt_length, 0)

    def test_collator_pads_tokens_but_never_masks_attention_for_real_tokens(self) -> None:
        collator = CausalLMCollator(pad_token_id=0)
        batch = collator(
            [
                {"input_ids": [10, 11], "labels": [-100, 11]},
                {"input_ids": [20], "labels": [20]},
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[10, 11], [20, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 11], [20, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_overlong_sequences_are_removed_instead_of_truncated(self) -> None:
        tokenizer = _CharacterTokenizer()
        dataset = Dataset.from_dict(
            {
                "name": ["func0"],
                "asm": ["mov $1, %eax\nretq"],
                "code": ["int func0(void) { return 1; }"],
                "file": ["project/src/func0.c"],
            }
        )
        tokenized = tokenize_dataset(
            dataset,
            cast(PreTrainedTokenizerBase, tokenizer),
            max_seq_length=8,
        )
        self.assertEqual(len(tokenized), 0)


if __name__ == "__main__":
    unittest.main()
