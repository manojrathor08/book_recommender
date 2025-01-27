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
- source_sentence: ' explains what’s wrong with our approach towards happiness and
    gives philosophical suggestions that help us make our lives worth living.'
  sentences:
  - ' is a modern, philosophic take on the joys of going away, exploring why we do
    so in the first place and how we can avoid falling into today’s most common tourist
    traps.'
  - ' is the blueprint you need to turn your passion into your profession and will
    give you the tools to turn yourself into a brand, leverage social media, produce
    great content and reap the financial benefits of it.'
  - ' shows why you should quit social media because it stops joy, makes you a jerk,
    erodes truth, kills empathy, takes free will, keeps the world insane, destroys
    authenticity, blocks economic dignity, makes politics a mess, and hates you.'
- source_sentence: ' is a multidisciplinary study that employs anthropological, biological,
    evolutionary, and socio-economic analyses to chart the fates of different peoples
    throughout human history and understand why some groups succeeded to develop and
    advance, while others haven’t.'
  sentences:
  - ' explores the philosophy of life and the secrets behind peak performance in MMA
    of John Kavanagh, the trainer and friend of superstar Conor McGregor, and their
    journey to success which started in a modest gym in Ireland and ended up with
    McGregor having a net worth of 100 million dollars. '
  - ' explores the curious mind of man’s best friend in relation to human intelligence,
    as dogs and humans are connected and have many similarities that make the relationship
    between them so strong and unique. '
  - ' teaches you how to beat overthinking by challenging whether your thoughts are
    true, retiring unhelpful and unkind ideas, adopting thought-boosting mantras from
    others, using symbols to reinforce positive thoughts, and more.'
- source_sentence: ' teaches you countless principles to become a likable person,
    handle your relationships well, win others over and help them change their behavior
    without being intrusive.'
  sentences:
  - ' shows you the surprisingly big influence other people have on your life, what
    different kinds of relationships you have with them and how you can cultivate
    more good ones to replace the bad, fake or unconnected and live a more fulfilled
    life.'
  - ' describes a scientific approach to being happier by giving you a short quiz
    to determine your “happiness set point,” followed by various tools and tactics
    to help you take control of the large chunk of happiness that’s fully within your
    grasp.'
  - ' will help you become a better leader in the office by sharing the life and teachings
    of businessman Bill Campbell who helped build multi-billion dollar companies in
    Silicon Valley.'
- source_sentence: ' shows how much of what’s truly important in life can be solved
    by the wisdom left behind by brilliant minds from long past. '
  sentences:
  - ' provides a more grounded way of living by eliminating the cult of being productive
    all the time to achieve success, instead offering a way to be at peace with yourself,
    prioritizing mental health and a simple yet meaningful life. '
  - ' will help you make everything, even the worst of times, go more smoothly by
    learning about a few useful phrases to habitually use come rain or shine.'
  - ' is the result of a 7-day meeting between the Dalai Lama and Desmond Tutu, two
    of the world’s most influential spiritual leaders, during which they discussed
    one of life’s most important questions: how do we find joy despite suffering?'
- source_sentence: ' takes a close look at the life of Albert Einstein, beginning
    in how his childhood shaped him, what his biggest discoveries and personal struggles
    were and how his focus changed in later years, without his genius ever fading
    until his very last moment.'
  sentences:
  - ' helps you understand why habits are at the core of everything you do, how you
    can change them, and what impact that will have on your life, your business and
    society.'
  - ' will help you become healthier by teaching you the truth behind the mind-body
    connection, revealing how your mental state does in fact affect your physical
    condition and how you can improve both.'
  - ' shows how people and companies can adapt in the rapidly changing world we live
    in today, explaining how a growth mindset, colleaboration, and losing your ego
    will build your confidence that you can stay relevant and competitive as the world
    around you accelerates.'
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
    ' takes a close look at the life of Albert Einstein, beginning in how his childhood shaped him, what his biggest discoveries and personal struggles were and how his focus changed in later years, without his genius\xa0ever fading until his very last moment.',
    ' shows how people and companies can adapt in the rapidly changing world we live in today, explaining how a growth mindset, colleaboration, and losing your ego will build your confidence that you can stay relevant and competitive as the world around you accelerates.',
    ' helps you understand why\xa0habits are at the core of everything you\xa0do, how you can change them, and what impact that will have on your life, your business and society.',
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
  |         | sentence_0                                                                         | sentence_1                                                                       | label                                                             |
  |:--------|:-----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                           | float                                                             |
  | details | <ul><li>min: 12 tokens</li><li>mean: 41.11 tokens</li><li>max: 67 tokens</li></ul> | <ul><li>min: 2 tokens</li><li>mean: 41.0 tokens</li><li>max: 70 tokens</li></ul> | <ul><li>min: -0.07</li><li>mean: 0.36</li><li>max: 0.79</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                              | sentence_1                                                                                                                                                                                                                         | label                            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code> shows you how to stop looking at the clock and start looking at the compass, by figuring out what’s important, prioritizing those things in your life, developing a vision for the future, building the right relationships and becoming a strong leader wherever you go.</code> | <code> teaches us how to persist in creative work when our brain wants to take a million different paths, showing us how to harness our brain power in moments of innovation as well as tediousness.</code>                        | <code>0.41934635043144225</code> |
  | <code> shares James Comey’s experiences as the director of the FBI and outlines what he learned about leadership, ethics, and politics throughout his life, career, and experiences with President Trump who was the reason he lost his job in May of 2017.</code>                      | <code> will help you become more patient with the speed of your progress by identifying the damaging influences of early achievement culture and societal pressure and how to be proud of reaching your peak later in life.</code> | <code>0.06931390687823295</code> |
  | <code> gives the relationship advice of hundreds of couples who have stayed together into old age and will teach you how to have happiness and longevity in your love life.</code>                                                                                                      | <code> shows you how to start a successful business based on the principles of the founders of some of the world’s most famous and accomplished startups.</code>                                                                   | <code>0.18417405635118486</code> |
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