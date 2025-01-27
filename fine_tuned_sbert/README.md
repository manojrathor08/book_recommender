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
- source_sentence: ' contains Oprah Winfrey’s tips for how to discover your real purpose
    so you can live a life of success and significance.'
  sentences:
  - ' shows you how to align your life with your most important goals, by finding
    out what’s really essential, changing your habits one at a time and working focused
    and productively on only those projects that will lead you to where you really
    want to go.'
  - ' contains the awful story of murder amid the Northern Ireland Conflict and a
    reflection on what caused it, those who were primarily involved, some of the worst
    parts of what happened, and other details of this dark era in the history of Ireland.'
  - ' is a refreshing, fun look at personal finance, that takes away the feeling that
    financial planning is a burden for the less disciplined, and shows you that you
    can plan your entire financial future on a single page.'
- source_sentence: ' will revolutionize your thinking with questions that create a
    learning mindset. '
  sentences:
  - ' will teach you how to lead others with lasting influence by focusing on your
    people instead of your position.'
  - ' is a compilation of laws that provides insights for conducting successful marketing
    campaigns by focusing on the essence of branding and how brands must be created
    and managed in order to survive in the competitive world.'
  - ' outlines the future of technology by describing how change keeps accelerating,
    what computers will look like and be made of, why biology and technology will
    become indistinguishable and how we can’t possibly predict what’ll happen after
    2045.'
- source_sentence: ' is based on the idea that we believe whatever we want to believe,
    and that it’s exactly this trait of ours, which marketers use (and sometimes abuse)
    to sell their products by infusing them with good stories – whether they’re true
    or not.'
  sentences:
  - ' gives powerful inspiration to change your life by helping you identify what
    you should improve on, how to get over the hurdles in your way, and the patterns
    and habits you need to set so that achieving your dreams is more possible than
    ever.'
  - ' is a message to everyone who’s not on the social media train yet, showing them
    how to tell their story the right way on social media, so that it’ll actually
    get heard.'
  - ' is your guide for learning how to stop pushing yourself to do more at your job
    and live a happier and more fulfilling life by making your money work hard for
    you. '
- source_sentence: ' gives you advice to declutter your space and keep it orderly,
    to foster your inner peace and allow you to flourish.'
  sentences:
  - ' brings together the spiritual calmness and mindful behavior of Eastern religions
    with Western striving for achieving internal and external success, showing you
    seven specific ways to let both come to you.'
  - ' will show you how to harness the power of slowing down your body and mind for
    less distractions, better self-control, and, above all, a happier and more peaceful
    life.'
  - ' is a historical exploration of the four primary elements we use to transform
    our food, from fire to water, air, and earth, celebrating traditional cooking
    methods while showing you practical ways to improve your eating habits and prepare
    more of your own food.'
- source_sentence:  is a story-based, stern yet entertaining self-help manual for
    young people laying out a set of simple rules to help us become more disciplined,
    behave better, act with integrity, and balance our lives while enjoying them as
    much as we can.
  sentences:
  - ' explains why people with many talents don’t fit into a world where we need specialists
    and, if you have many talents yourself, shows you how you can lift this curse,
    by giving you a framework to follow and find your true vocation in life.'
  - ' teaches you the 10 qualities of winners, which set them apart and help them
    win in every sphere of life: personally, professionally and spiritually.'
  - ' will make you smarter and healthier by teaching you about the tiny ecosystems
    of microbes that live inside your body and on everything you see and by showing
    you how they affect your life and how to utilize them to improve your well-being.'
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
    '\xa0is a story-based, stern yet entertaining self-help manual for young people laying out a set of simple rules to help us become more disciplined, behave better, act with integrity, and balance our lives while enjoying them as much as we can.',
    ' will make you smarter and healthier by teaching you about the tiny ecosystems of microbes that live inside your body and on everything you see and by showing you how they affect your life and how to utilize them to improve your well-being.',
    ' explains why people with many talents don’t fit into a world where we need specialists and, if you have many talents yourself, shows you how you can lift this curse, by giving you a framework to follow and find your true vocation in life.',
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
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                            |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                            |
  | details | <ul><li>min: 2 tokens</li><li>mean: 41.04 tokens</li><li>max: 73 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 41.58 tokens</li><li>max: 81 tokens</li></ul> | <ul><li>min: -0.04</li><li>mean: 0.36</li><li>max: 0.9</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                    | sentence_1                                                                                                                                                                                                                                  | label                            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code> will enlarge your knowledge of our expanding universe by showing you how it began, what we’re learning about it now, and what will happen to it in the future.</code>                  | <code> draws on many of history’s most famous power quarrels to show you what power looks like, how you can get it, what to do to defend yourself against the power of others and, most importantly, how to use it well and keep it.</code> | <code>0.11771032512187958</code> |
  | <code> is a blueprint to help you close the gap between your day job and your dream job, showing you simple steps you can take towards your dream without turning it into a nightmare.</code> | <code> contains Oprah Winfrey’s tips for how to discover your real purpose so you can live a life of success and significance.</code>                                                                                                       | <code>0.3589766651391983</code>  |
  | <code> explains why Silicon Valley is suffering a nervous breakdown as big data and machine intelligence comes to an end and the post-Google era dawns.</code>                                | <code> makes you more vigilant of the warning signs of oppression by identifying it’s political nature, how to protect yourself and society, and what you can do to resist dangerous leadership.</code>                                     | <code>0.137077134847641</code>   |
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