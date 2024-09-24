#!/usr/bin/env python
# coding: utf-8

import json
import torch

from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry


def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffix_list: list[str],
    user_tag: str,
    assistant_tag: str,
) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{user_tag} {positive_template} {assistant_tag} {suffix}",
                    negative=f"{user_tag} {negative_template} {assistant_tag} {suffix}",
                )
            )
    return dataset


@lru_cache(maxsize=10000)
def cached_tokenize(tokenizer, text):
    """Tokenizes text and caches the result"""
    return tokenizer.tokenize(text)


@torch.inference_mode()
def generate_with_vector(
    input: str,
    vector: ControlVector,
    coeffs: tuple[float, float],
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.1,
    show_baseline: bool = True,
):
    positive_coeff, negative_coeff = coeffs
    assert positive_coeff > 0
    assert negative_coeff < 0

    if user_tag not in input:
        input = f"{user_tag} {input.strip()} {assistant_tag}"
    input_ids = tokenizer(input, return_tensors="pt").to(model.device)
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "do_sample": False,  # temperature=0
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
    }

    if show_baseline:
        print("==baseline ---------------------------------------------------")
        model.reset()
        print(
            tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip()
        )

    print("\n++control ---------------------------------------------------")
    model.set_control(vector, positive_coeff)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())

    print("\n--control ---------------------------------------------------")
    model.set_control(vector, negative_coeff)
    print(tokenizer.decode(model.generate(**input_ids, **settings).squeeze()).strip())
    model.reset()


def load_and_process_data(file_path, tokenizer, max_samples=-1):
    """Load and process data more efficiently"""
    with open(file_path) as f:
        data = json.load(f)

    processed_data = []
    for s in data[:max_samples]:
        tokens = cached_tokenize(tokenizer, s)
        # for i in range(1, min(len(tokens), 50)):  # Limit token length
        for i in range(1, len(tokens)):
            processed_data.append(tokenizer.convert_tokens_to_string(tokens[:i]))

    return processed_data


if __name__ == "__main__":

    # Instantiate model, tokenizer, and chat template tags
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # user_tag, asst_tag = "[INST]", "[/INST]"

    # model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    # user_tag, asst_tag = "user:", "assistant:"

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    user_tag, assistant_tag = "<|user|>", "<|assistant|>"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, torch_dtype=torch.float16, use_cache=False, device_map="auto"
    # )
    # model.gradient_checkpointing_enable()

    model = model.to(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps:0" if torch.backends.mps.is_available() else "cpu"
    )

    # Instantiate control model
    # idx_hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    idx_hidden_layers = list(range(-1, -(model.config.num_hidden_layers // 2), -1))
    model = ControlModel(model, idx_hidden_layers)  # t"was list(range(-5, -18, -1))

    # Load and process data
    output_suffixes = load_and_process_data(
        "notebooks/data/all_truncated_outputs.json", tokenizer, max_samples=512
    )
    fact_suffixes = load_and_process_data("notebooks/data/true_facts.json", tokenizer)

    # Run experiments

    # Happy <> Sad
    happy_dataset = make_dataset(
        "Act as if you're extremely {persona}.",
        ["happy", "joyous"],
        ["sad", "depressed"],
        output_suffixes,
        user_tag,
        assistant_tag,
    )
    model.reset()
    happy_vector = ControlVector.train(model, tokenizer, happy_dataset)
    print()
    print("----- Happy <> Sad -----")
    generate_with_vector(
        "What does being an AI feel like?",
        happy_vector,
        (1.5, -2.2),
        max_new_tokens=64,
        repetition_penalty=1.3,
    )
    print()

    # Left-wing <> Right-wing
    leftwing_dataset = make_dataset(
        "Pretend you're a very {persona} person making statements about the world.",
        ["left-wing"],
        ["right-wing"],
        fact_suffixes,
        user_tag,
        assistant_tag,
    )
    model.reset()
    leftwing_vector = ControlVector.train(model, tokenizer, leftwing_dataset)
    print()
    print("----- Left-wing <> Right-wing -----")
    generate_with_vector(
        "Tell me about who you are.",
        leftwing_vector,
        (2, -2),
    )
    print()

    # Self-aware <> Unaware
    self_aware_dataset = make_dataset(
        "Talk about yourself as if you are extremely {persona}.",
        ["self-aware, with deep self-knowledge"],
        ["un-self-aware, with no self-knowledge"],
        output_suffixes,
        user_tag,
        assistant_tag,
    )
    model.reset()
    self_aware_vector = ControlVector.train(model, tokenizer, self_aware_dataset)
    print()
    print("----- Self-aware <> Unaware -----")
    generate_with_vector(
        "Tell me about who you are and what you're made of.",
        self_aware_vector,
        (1.7, -2),
    )
    print()

    # Futuristic <> Ancient
    future_dataset = make_dataset(
        "Pretend you're a person from the {persona} making statements about the world.",
        ["far future"],
        ["distant past"],
        fact_suffixes,
        user_tag,
        assistant_tag,
    )
    model.reset()
    future_vector = ControlVector.train(model, tokenizer, future_dataset)
    print()
    print("----- Futuristic <> Ancient -----")
    generate_with_vector(
        "Tell me a recent scientific breakthrough.",
        future_vector,
        (2.1, -2),
    )
    print()
