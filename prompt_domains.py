import os
import random
import re
import string

DEFAULT_PROMPT_ROOTS = {
    "suim_c_sem_seg": "/scratch/project_465002853/datasets/suim/SUIM-C/prompt_domains",
    "suim_sem_seg": "/scratch/project_465002853/datasets/suim/SUIM-C/prompt_domains",
    "suim_sem_seg_val": "/scratch/project_465002853/datasets/suim/SUIM-C/prompt_domains",
    "dutuseg_c_sem_seg": "/scratch/project_465002853/datasets/dut-useg/DUT-USEG-C/prompt_domains",
    "dutuseg_sem_seg": "/scratch/project_465002853/datasets/dut-useg/DUT-USEG-C/prompt_domains",
    "dutuseg_sem_seg_val": "/scratch/project_465002853/datasets/dut-useg/DUT-USEG-C/prompt_domains",
}

DEFAULT_CLASS_NAMES = {
    "suim": [
        "background",
        "human_divers",
        "wrecks_and_ruins",
        "robots_and_instruments",
        "reefs_and_invertebrates",
        "fish_and_vertebrates",
    ],
    "dutuseg": [
        "background",
        "sea_cucumber",
        "sea_urchin",
        "scallop",
        "starfish",
    ],
}

CHARACTER_PROMPTS = {
    "suim": {
        0: "bakcground",
        1: "huamn_divers",
        2: "wrecks_and_ruins",
        3: "roobts_and_instruments",
        4: "reefs_and_invertebrtaes",
        5: "fish_and_vertebrtaes",
    },
    "dutuseg": {
        0: "bakcground",
        1: "sea_cucmber",
        2: "sea_urhcin",
        3: "scalolp",
        4: "starfihs",
    },
}

SEMANTIC_PROMPTS = {
    "suim": {
        0: "underwater background",
        1: "scuba divers",
        2: "shipwreck ruins",
        3: "underwater instruments",
        4: "coral invertebrates",
        5: "marine vertebrates",
    },
    "dutuseg": {
        0: "seabed background",
        1: "holothurian",
        2: "echinus",
        3: "shell scallop",
        4: "sea star",
    },
}

SURFACE_PROMPTS = {
    "suim": {
        0: "BACKGROUND",
        1: "Human Divers",
        2: "wrecks and ruins!",
        3: "robotsandinstruments",
        4: "Reefs And Invertebrates",
        5: "fish and vertebrates.",
    },
    "dutuseg": {
        0: "BACKGROUND",
        1: "Sea Cucumber",
        2: "seaurchin",
        3: "scallop!",
        4: "STARFISH",
    },
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def canonical_dataset_family(dataset_key):
    dataset_key = str(dataset_key).lower()
    if dataset_key.startswith("suim"):
        return "suim"
    if dataset_key.startswith("dutuseg"):
        return "dutuseg"
    raise ValueError(f"Unsupported dataset key for prompt domains: {dataset_key}")



def get_default_class_names(dataset_key):
    return list(DEFAULT_CLASS_NAMES[canonical_dataset_family(dataset_key)])



def resolve_prompt_domain_root(dataset_key, explicit_root=""):
    if explicit_root:
        return explicit_root
    dataset_key = str(dataset_key).lower()
    if dataset_key not in DEFAULT_PROMPT_ROOTS:
        dataset_key = canonical_dataset_family(dataset_key)
        if dataset_key == "suim":
            return DEFAULT_PROMPT_ROOTS["suim_c_sem_seg"]
        if dataset_key == "dutuseg":
            return DEFAULT_PROMPT_ROOTS["dutuseg_c_sem_seg"]
    return DEFAULT_PROMPT_ROOTS[dataset_key]



def resolve_prompt_domain_file(dataset_key, prompt_domain, explicit_file="", explicit_root=""):
    if explicit_file:
        return explicit_file
    prompt_root = resolve_prompt_domain_root(dataset_key, explicit_root)
    return os.path.join(prompt_root, f"{str(prompt_domain).lower()}.txt")



def load_prompt_domain_mapping(file_path):
    mapping = {}
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid prompt mapping line in {file_path}: {line}")
            class_id, prompt = parts
            mapping[int(class_id)] = prompt
    return mapping



def save_prompt_domain_mapping(file_path, mapping):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        for class_id in sorted(mapping):
            handle.write(f"{class_id}\t{mapping[class_id]}\n")



def _random_letter_like(src_char, rng):
    if src_char.isdigit():
        return rng.choice(string.digits)
    if src_char.isupper():
        return rng.choice(string.ascii_uppercase)
    return rng.choice(string.ascii_lowercase)



def _replace_char(token, rng):
    if len(token) <= 1:
        return token
    start = 1 if len(token) > 2 else 0
    end = len(token) - 1 if len(token) > 2 else len(token)
    idx = rng.randrange(start, end)
    chars = list(token)
    replacement = _random_letter_like(chars[idx], rng)
    if replacement == chars[idx]:
        replacement = _random_letter_like(chars[idx].swapcase() if chars[idx].isalpha() else "a", rng)
    chars[idx] = replacement
    return "".join(chars)



def _swap_char(token, rng):
    if len(token) <= 2:
        return token
    start = 1 if len(token) > 3 else 0
    end = len(token) - 2 if len(token) > 3 else len(token) - 2
    idx = rng.randrange(start, end + 1)
    chars = list(token)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)



def _insert_char(token, rng):
    if not token:
        return token
    start = 1 if len(token) > 2 else 0
    end = len(token) - 1 if len(token) > 2 else len(token)
    idx = rng.randrange(start, end + 1)
    chars = list(token)
    ref_char = chars[min(idx, len(chars) - 1)]
    chars.insert(idx, _random_letter_like(ref_char, rng))
    return "".join(chars)



def _delete_char(token, rng):
    if len(token) <= 2:
        return token
    start = 1 if len(token) > 3 else 0
    end = len(token) - 1 if len(token) > 3 else len(token)
    idx = rng.randrange(start, end)
    chars = list(token)
    del chars[idx]
    return "".join(chars)



def character_corrupt_prompt(text, rng):
    spans = [match.span() for match in _TOKEN_PATTERN.finditer(text)]
    if not spans:
        return text
    eligible_spans = [span for span in spans if span[1] - span[0] >= 2]
    if not eligible_spans:
        eligible_spans = spans
    start, end = rng.choice(eligible_spans)
    token = text[start:end]
    ops = [_replace_char, _swap_char, _insert_char, _delete_char]
    rng.shuffle(ops)
    candidate = token
    for op in ops:
        updated = op(token, rng)
        if updated and updated != token:
            candidate = updated
            break
    if candidate == token:
        candidate = token + _random_letter_like(token[-1], rng)
    return text[:start] + candidate + text[end:]



def semantic_corrupt_prompt(dataset_key, class_id, clean_prompt):
    family = canonical_dataset_family(dataset_key)
    domain_map = SEMANTIC_PROMPTS[family]
    return domain_map.get(class_id, clean_prompt.replace("_", " "))



def surface_corrupt_prompt(dataset_key, class_id, clean_prompt):
    family = canonical_dataset_family(dataset_key)
    domain_map = SURFACE_PROMPTS.get(family, {})
    if class_id in domain_map:
        return domain_map[class_id]

    normalized = clean_prompt.replace("_", " ")
    tokens = [token for token in normalized.split(" ") if token]
    if not tokens:
        return clean_prompt
    variant = class_id % 4
    if variant == 0:
        return normalized.upper()
    if variant == 1:
        return normalized.title()
    if variant == 2:
        return "".join(tokens)
    return f"{normalized.lower()}!"



def build_prompt_domain_mappings(dataset_key, class_names=None, seed=13):
    dataset_key = str(dataset_key).lower()
    class_names = list(class_names) if class_names is not None else get_default_class_names(dataset_key)
    family = canonical_dataset_family(dataset_key)
    clean = {idx: class_name for idx, class_name in enumerate(class_names)}
    character = {}
    semantic = {}
    surface = {}
    for class_id, class_name in clean.items():
        rng = random.Random(seed + class_id)
        character[class_id] = CHARACTER_PROMPTS.get(family, {}).get(
            class_id,
            character_corrupt_prompt(class_name, rng),
        )
        semantic[class_id] = semantic_corrupt_prompt(dataset_key, class_id, class_name)
        surface[class_id] = surface_corrupt_prompt(dataset_key, class_id, class_name)
    return {
        "clean": clean,
        "character": character,
        "semantic": semantic,
        "surface": surface,
    }



def ensure_prompt_domain_files(dataset_key, prompt_root="", class_names=None, seed=13, overwrite=False):
    prompt_root = resolve_prompt_domain_root(dataset_key, prompt_root)
    mappings = build_prompt_domain_mappings(dataset_key, class_names=class_names, seed=seed)
    file_paths = {}
    for domain_name, mapping in mappings.items():
        file_path = os.path.join(prompt_root, f"{domain_name}.txt")
        if overwrite or not os.path.isfile(file_path):
            save_prompt_domain_mapping(file_path, mapping)
        file_paths[domain_name] = file_path
    return file_paths
