---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:15000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: ' makes you smarter by helping you understand where this important
    aspect of our lives comes from, how we’ve used it throughout history to get to
    where we are today, and why we need to be careful about how we consume it so that
    we can have a better future.'
  sentences:
  - ' makes you smarter and more compassionate by revealing the previously unknown
    story of a woman with extraordinary cells that still live today and have contributed
    to dozens of medical breakthroughs.'
  - ' shows you how to become your best self and live up to your full potential by
    outlining nine science-backed ways to beat the odds and achieve your goals and
    dreams.'
  - ' is a classic novel from 1605 which portraits the life and insightful journey
    of Don Quixote de la Mancha, a Spanish man who seems to be losing his mind on
    his quest to become a knight and restore chivalry alongside with a farmer named
    Sancho Panza, with whom he fights multiple imaginary enemies and faces a series
    of fantastic challenges.'
- source_sentence: ' explains the importance of emotions in your life, how they help
    and hurt your ability to navigate the world, followed by practical advice on how
    to improve your own emotional intelligence and why that is the key to leading
    a successful life.'
  sentences:
  - ' identifies the hidden superpowers of introverts and empowers them by helping
    them understand why it’s so difficult to be quiet in a world that’s loud and how
    to ease their way into becoming confident in social situations.'
  - ' uses science, spirituality, humor, and Dante’s Divine Comedy to teach you how
    to find well-being, healing, a sense of purpose, and much more by rediscovering
    integrity, or the recently lost art of living true to yourself by what you do,
    think and say.'
  - ' will improve your mental state and level of success by identifying what you
    get wrong about joy and how to discover what’s most important to you and how to
    make those things a more significant part of your life.'
- source_sentence: ' will help you become happier, find your purpose, overcome your
    fears, and begin living the life you’ve always wanted by identifying the steps
    you need to take to connect with a higher spiritual power.'
  sentences:
  - ' is an international bestseller that will help you unearth your sad, suppressed
    memories from childhood that still haunt you today and teach you how to confront
    them so you can avoid passing them on to your children, release yourself from
    the pains of your past, and finally be free to live a life of fulfillment.'
  - ' advocates against the use of sugar in the food industry and offers a critical
    look at how this harmful substance took over the world under the eyes of our highest
    institutions, who are very well aware of its toxicity but choose to remain silent.'
  - ' is about changing your overall perspective, so you can embrace a philosophy
    that’ll help you achieve your full potential in work, relationships, finance,
    and all other walks of life.'
- source_sentence: ' by Michael Newman outlines the history of the governmental theory
    that everything should be owned and controlled by the community as a whole, including
    how this idea has impacted the world in the last 200 years, how its original aims
    have been lost, and ways we might use it in the future.'
  sentences:
  - ' is a collection of essays by Y Combinator founder Paul Graham about what makes
    a good computer programmer and how you can code the future if you are one, making
    a fortune in the process.'
  - ' offers its readers a focus-based approach that they can use to achieve their
    financial and personal goals through practical exercises and habits that they
    can implement into their daily lives to actively shape their future.'
  - ' offers a hands-on guide to living a meaningful life and letting go of negative
    thoughts by compiling the groundbreaking theories of psychologist Alfred Adler
    with other valuable research into an all-in-one book for becoming a happy and
    fulfilled person.'
- source_sentence: ' is a collection of actionable tips to help you master the art
    of human communication, leave great first impressions and make people feel comfortable
    around you in all walks of life.'
  sentences:
  - ' helps leaders act based on the future, not the past, and allows them to create
    organizational change at a global level through creative and agile methodologies. '
  - ' gives you the confidence to make it through life’s inevitable setbacks by sharing
    ideas and strategies like mindfulness to grow your resilience and come out on
    top.'
  - ' is a life-changing guide to growing your self-confidence that shows how posture,
    mindset, and body language all expand your feeling of empowerment and your communication
    skills.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    ' is a collection of actionable tips to help you master the art of human communication, leave great first impressions and make people feel comfortable around you in all walks of life.',
    ' is a life-changing guide to growing your self-confidence that shows how posture, mindset, and body language all expand your feeling of empowerment and your communication skills.',
    ' helps leaders act based on the future, not the past, and allows them to create organizational change at a global level through creative and agile methodologies. ',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 15,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                        | label                                                             |
  |:--------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                            | float                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 40.8 tokens</li><li>max: 67 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 41.9 tokens</li><li>max: 81 tokens</li></ul> | <ul><li>min: -0.03</li><li>mean: 0.36</li><li>max: 0.76</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                            | sentence_1                                                                                                                                                                                                                                                                  | label                           |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------|
  | <code> identifies the stumbling blocks that are in your way of reaching your goals and improving yourself and the research-backed ways to get over them, including how to beat some of the worst productivity and life problems like procrastination, laziness, and much more.</code> | <code> is a classic self-improvement book that will boost your happiness and give you the life of your dreams by identifying what Napoleon Hill learned interviewing hundreds of successful people and sharing how their outlook on life helped them get to the top.</code> | <code>0.3772610414028168</code> |
  | <code> teaches how you can become more content and happy in your life by applying the principles of meditation and Buddhism. </code>                                                                                                                                                  | <code> will show you how to harness the power of slowing down your body and mind for less distractions, better self-control, and, above all, a happier and more peaceful life.</code>                                                                                       | <code>0.5244289985724858</code> |
  | <code> is a reminder to slow down and learn to appreciate the little moments in life, like the times when we’re really just waiting for the next big thing, as they shape our lives a lot more than we think.</code>                                                                  | <code> outlines the importance of doing small, little improvements in our everyday life to achieve a successful bigger picture, and how by focusing more on making better day-by-day choices you can shape a remarkable future.</code>                                      | <code>0.5583940505981446</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 4.2373 | 500  | 0.0011        |
| 8.4746 | 1000 | 0.0008        |


### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 3.3.1
- Transformers: 4.47.1
- PyTorch: 2.5.1+cu121
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->