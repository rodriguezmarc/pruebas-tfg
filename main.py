from datasets import load_dataset

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor
)

from peft import LoraConfig, get_peft_model
from aux import collate_fn

###### ---------- PREPROCESS ------------- ######

ds = load_dataset("food101")  # Food-101 contains images of 101 food classes

# as each food is labeled with an integer, create dictionary to map to label
labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# load image processor to properly resize and normalize pixel values
# IMAGE PROCESSOR: loads, preprocesses input features and preprocesses outputs
# includes normalization, resizing, conversion, etc.
# Will be used to obtain precomputed values of great value to the training
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# prepare transformation functions 
normalize = Normalize(                # normalization function
    mean=image_processor.image_mean,
    std=image_processor.image_std
)

# COMPOSE: composes several transforms together

train_transforms = Compose(           # train set transformation pipeline
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(batch):
    batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in batch["image"]]
    return batch

val_transforms = Compose(             # validation transformation pipeline
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_val(batch):
    batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in batch["image"]]
    return batch


# Use .set_transform() to apply transformations on-the-fly
# SET_TRANSFORM: returns format using specified transform, applied on batches when called
# - tranform: Callable: user-defined formatting, takes a batch as a dict and returns a batch
# - columns: List[str]: columns to format in the output
# - output_all_columns: bool: output keeps un-formatted columns as well
train_ds = ds["train"]
train_ds.set_transform(transform=preprocess_train)

val_ds = ds["validation"]
val_ds.set_transform(transform=preprocess_val)

###### ---------- MODEL CONFIG ------------- ######

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

# Every PEFT method requires config that holds params specifying how should be applied
# once it is setup, pass it to the get_peft_model() along with base model to create trainable PeftModel

# LORACONFIG: base configuration class to store config of a PeftModel
# - r: int: lora attention dimension or rank
# - lora_alpha: int: param for lora scaling
# - lora_dropout: float: dropout probability for lora
# - target_modules: Union[List[str], str]: names of modules to apply the adapter to (be replaced)
# - bias: str: bias type for lora
# - modules_to_save: List[str]: list of modules apart from adapter layers to be set as trainable and saved
config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
    modules_to_save=["classifier"]
)

# PEFTMODEL: base model encompassing Peft methods
# - model: PreTrainedModel: base transformer model used for peft
# - peft_config: PeftConfig: configuration of the peft model
# - adapter_name: str: name of the adapter
model = get_peft_model(model, config)
model.print_trainable_parameters()

###### ---------- MODEL TRAINING ------------- ######

# let's use trainer class, contains trainig loop and when ready call train.
# to customize training run, configure hyperparameters in TrainingArguments
account="rodriguezmarc"
peft_model_id=f"{account}/google/vit-base-patch16-224-in21k-lora"
batch_size=1  # batch size config.

# TRAINING ARGUMENTS:
args = TrainingArguments(
    peft_model_id,                              # output dir
    remove_unused_columns=False,                # automatically remove unused columns
    eval_strategy="epoch",                      # when to run eval {steps=every eval_steps; epoch=at the end of each epoch}
    save_strategy="epoch",                      # checkpoint save strategy to adopt during training
    learning_rate=5e-3,                         # initial lr for optimizer

    per_device_train_batch_size=batch_size,     # train batch size per device in multi-GPU
    per_device_eval_batch_size=batch_size,      # eval batch size per device in multi-GPU
    gradient_accumulation_steps=4,              # number of update steps to accumulate gradients before backward pass

    fp16=False,                                 # float16 mixed precision training
    dataloader_pin_memory=False,                # want to pin memory to data loaders or not
    dataloader_num_workers=0,                   # num of subprocesses to use for data loading

    num_train_epochs=5,                         # num of training epochs to perform
    logging_steps=10,                           # num of update steps between two logs 
    load_best_model_at_end=True,                # load best checkpoint at the end of training
    label_names=["labels"],                     # dict inputs that correspond to labels
)

# TRAINER: feature-complete training and eval loop optimized for transformers
trainer = Trainer(
    model,                              # peft model
    args,                               # train arguments
    train_dataset=train_ds,             # train dataset
    eval_dataset=val_ds,                # evaluation dataset
    processing_class=image_processor,   # used to process
    data_collator=collate_fn            # used to form a barch from ds
)
trainer.train()

from huggingface_hub import notebook_login
notebook_login()
model.push_to_hub(peft_model_id)