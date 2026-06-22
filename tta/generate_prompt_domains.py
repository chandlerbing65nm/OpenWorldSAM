import argparse

from prompt_domains import ensure_prompt_domain_files, get_default_class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["suim_c_sem_seg", "dutuseg_c_sem_seg", "suim_sem_seg_val", "dutuseg_sem_seg_val"],
        required=False,
        help="Generate prompt-domain txt files for a single dataset.",
    )
    parser.add_argument(
        "--prompt-root",
        default="",
        help="Optional override directory for output txt files.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["suim_c_sem_seg", "dutuseg_c_sem_seg"]
    for dataset_name in datasets:
        file_paths = ensure_prompt_domain_files(
            dataset_name,
            prompt_root=args.prompt_root,
            class_names=get_default_class_names(dataset_name),
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(f"[{dataset_name}]")
        for domain_name in ("clean", "character", "semantic", "surface"):
            print(f"  {domain_name}: {file_paths[domain_name]}")


if __name__ == "__main__":
    main()
