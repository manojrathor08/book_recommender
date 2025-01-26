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
- source_sentence:  claims that everything you think you know about managing people
    is wrong, revealing how you can challenge the status quo so that both you and
    those you lead will achieve their full potential.
  sentences:
  - ' explains why you need manage your career as if you were running a start-up to
    get ahead in today’s ultra-competitive and ever-changing business world. '
  - ' explores the philosophy of eating according to your body’s needs and ditching
    diets, eating trends, and other limiting eating programs in favor of a well-balanced
    lifestyle built on personal body-related needs.'
  - ' shows you why the time of simply following instructions at your job is over
    and how to make yourself indispensable, which is a must for success today.'
- source_sentence: ' explores the curious mind of man’s best friend in relation to
    human intelligence, as dogs and humans are connected and have many similarities
    that make the relationship between them so strong and unique. '
  sentences:
  - ' teaches you how to be resourceful and prepare ahead of time for a world in which
    people not only live longer but reach an age in the triple-digits, and talks about
    what you should be doing right now to ensure you have enough money for retirement.'
  - ' is an early internet entrepreneurs step-by-step blueprint to creating products
    people want, launching them from the comfort of your home and building the life
    you’ve always wanted, thanks to the power of psychology, email, and of course
    the internet.'
  - ' describes recurring themes and trends throughout 5,000 years of human history,
    viewed through the lenses of 12 different fields, aimed at explaining the present,
    the future, human nature and the inner workings of states.'
- source_sentence: ' explores the effects of the Internet on the human brain, which
    aren’t entirely positive, as our constant exposure to the online environment through
    digital devices strips our ability to target our focus and stay concentrated,
    all while modifying our brain neurologically and anatomically. '
  sentences:
  - ' makes you smarter by helping you understand where this important aspect of our
    lives comes from, how we’ve used it throughout history to get to where we are
    today, and why we need to be careful about how we consume it so that we can have
    a better future.'
  - ' is a message to everyone who’s not on the social media train yet, showing them
    how to tell their story the right way on social media, so that it’ll actually
    get heard.'
  - ' is a look at how genes affect our abilities, motivations, and endurance in sports,
    explaining why some people are better suited for certain sports than others.'
- source_sentence: ' helps you understand and utilize the power of your mind-body
    connection by explaining the effect that your thoughts have on your body, including
    pain, illness, and memory and how to take advantage of it.'
  sentences:
  - ' tells you what you can do to overcome your negativity bias of focusing on and
    exaggerating negative events by relishing, extending and prioritizing the good
    things in your life to become happier.'
  - ' will improve your interpersonal and relationship skills by identifying the power
    of using mindfulness when talking with others, showing you how to listen with
    respect, convey your ideas efficiently, and most of all deepen your connections
    with others.'
  - ' shows you the surprisingly big influence other people have on your life, what
    different kinds of relationships you have with them and how you can cultivate
    more good ones to replace the bad, fake or unconnected and live a more fulfilled
    life.'
- source_sentence: ' shows you how to become a successful entrepreneur by explaining
    the steps necessary to grow a small service company and one day sell it.'
  sentences:
  - ' teaches how you can become more content and happy in your life by applying the
    principles of meditation and Buddhism. '
  - ' is about the hard-to-describe, yet powerful Danish attitude towards life, which
    consistently ranks Denmark among the happiest countries in the world and how you
    can cultivate it for yourself.'
  - ' teaches you how to craft the perfect business plan and grow your company by
    focusing on getting to know your customers and solving their problems then creating
    products to solve those issues.'
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
    ' shows you how to become a successful entrepreneur by explaining the steps necessary to grow a small service company and one day sell it.',
    ' teaches you how to craft the perfect business plan and grow your company by focusing on getting to know your customers and solving their problems then creating products to solve those issues.',
    ' teaches how you can become more content and happy in your life by applying the principles of meditation and Buddhism. ',
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
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                             |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                             |
  | details | <ul><li>min: 5 tokens</li><li>mean: 40.75 tokens</li><li>max: 67 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 41.91 tokens</li><li>max: 81 tokens</li></ul> | <ul><li>min: -0.07</li><li>mean: 0.36</li><li>max: 0.78</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                 | sentence_1                                                                                                                                                                                                                                                                       | label                            |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code> is a go-to guide that teaches us how to establish a mind-body-spirit connection and create better connections with the people around us by exploring how these aspects are interconnected and influenced by the way we eat, think, and feel.</code> | <code> delves into the subject of thinking mechanisms and cognitive processes, and explores how you can think more efficiently and draw better insights from the world around you by adopting a few key practices, such as filtering your thoughts or prioritizing work. </code> | <code>0.3919088661670685</code>  |
  | <code> shows you the three keys to arriving at work and life with a battery that’s brimming with happiness and motivation, which are energy, interactions and meaning, and how to implement them in your day.</code>                                       | <code> digs into neuroscientific research to explain why we’re not meant to multitask, how you can go back to the old, singletasking ways, and why that’s better for your work, relationships and happiness.</code>                                                              | <code>0.39517779690878735</code> |
  | <code> shows those in their twenties and thirties how to manage their finances so that they can stop scraping by and instead begin to live more confidently when it comes to money.</code>                                                                 | <code> is based on the assumption that the less you have to do, the more life you have to live, and helps you implement this philosophy into your life by giving you real-world tools to boost efficiency in every aspect of your life.</code>                                   | <code>0.3089342693487803</code>  |
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