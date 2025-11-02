from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import torch

weights = torch.tensor([1.0, 1.0])  # dummy example

loss_fn = CrossEntropyLoss(weight=weights)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

print("Everything imported successfully!")
