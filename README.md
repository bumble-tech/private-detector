# Private Detector

This is the repo for Bumble's *Private Detector*™ model - an image classifier that can detect lewd images.

The internal repo has been heavily refactored and released as a fully open-source project to allow for the wider community to use and finetune a Private Detector model of their own

## Model

The SavedModel can be found in `saved_model/`, ensure you have [Git LFS](https://git-lfs.github.com/) installed to pull the full files.

The model is based on Efficientnet-v2 and trained on our internal dataset of lewd images - more information can be found at the whitepaper [here](pending)

## Inference

Inference is pretty simple and an example has been given in `inference.py`

```sh
python3 inference.py \
    --model saved_model/ \
    --image_paths \
        Yes_samples/1.jpg \
        Yes_samples/2.jpg \
        Yes_samples/3.jpg \
        Yes_samples/4.jpg \
        Yes_samples/5.jpg \
        No_samples/1.jpg \
        No_samples/2.jpg \
        No_samples/3.jpg \
        No_samples/4.jpg \
        No_samples/5.jpg \
```

<details>
<summary>Sample Output</summary>
<code>

    Probability: 93.71% - Yes_samples/1.jpg
    Probability: 93.43% - Yes_samples/2.jpg
    Probability: 94.06% - Yes_samples/3.jpg
    Probability: 94.08% - Yes_samples/4.jpg
    Probability: 91.01% - Yes_samples/5.jpg
    Probability: 9.76% - No_samples/1.jpg
    Probability: 7.14% - No_samples/2.jpg
    Probability: 8.83% - No_samples/3.jpg
    Probability: 4.87% - No_samples/4.jpg
    Probability: 5.29% - No_samples/5.jpg
</code>
</details>

## Additional Training

You can finetune the model yourself on your own data, to do so is fairly simple - though you will need the checkpoint files as can be found in `saved_checkpoint/`

Set up a JSON file with links to your image path lists for each class:

```json
{
    "Yes": {
        "path": "/home/sofarrell/private_detector/Yes.txt",
        "label": 0
    },
    "No": {
         "path": "/home/sofarrell/private_detector/No.txt",
         "label": 1
    }
}
```

With each `.txt` file listing off the image paths to your images

```txt
/home/sofarrell/private_detector_images/Yes/1093840880_309463828.jpg
/home/sofarrell/private_detector_images/Yes/657954182_3459624.jpg
/home/sofarrell/private_detector_images/Yes/1503714421_3048734.jpg
```

You can create the training environment with conda:

```sh
conda env create -f environment.yaml
conda activate private_detector
```

And then retrain like so:

```sh
python3 ./train.py \
    --train_json /home/sofarrell/private_detector/train_classes.json \
    --eval_json /home/sofarrell/private_detector/eval_classes.json \
    --checkpoint_dir saved_checkpoint/ \
    --train_id retrained_private_detector
```

The training script has several parameters that can be tweaked:
|Command|Description|Type|Default|
|---|---|---|---|
|`train_id`|ID for this particular training run|str||
|`train_json`|JSON file(s) which describes classes and contains lists of filenames of data files|List[str]||
|`eval_json`|Validation json file which describes classes and contains lists of filenames of data files|str||
|`num_epochs`|Number of epochs to train for|int||
|`batch_size`|Number of images to process in a batch|int|`64`|
|`checkpoint_dir`|Directory to store checkpoints in|str||
|`model_dir`|Directory to store graph in|str|`.`|
|`data_format`|Data format: [channels_first, channels_last]|str|`channels_last`|
|`initial_learning_rate`|Initial learning rate|float|`1e-4`|
|`min_learning_rate`|Minimal learning rate|float|`1e-6`|
|`min_eval_metric`|Minimal evaluation metric to start saving models|float|`0.01`|
|`float_dtype`|Float Dtype to use in image tensors: [16, 32]|int|`16`|
|`steps_per_train_epoch`|Number of steps per train epoch|int|`800`|
|`steps_per_eval_epoch`|Number of steps per evaluation epoch|int|`1`|
|`reset_on_lr_update`|Whether to reset to the best model after learning rate update|bool|`False`|
|`rotation_augmentation`|Rotation augmentation angle, value <= 0 disables it|float|`0`|
|`use_augmentation`|Add speckle, v0, random or color distortion augmentation|str||
|`scale_crop_augmentation`|Resize image to the model's size times this scale and then randomly crop needed size|float|`1.4`|
|`reg_loss_weight`|L2 regularization weight|float|`0`|
|`skip_saving_epochs`|Do not save good checkpoint and update best metric for this number of the first epochs|int|`0`|
|`sequential`|Use sequential run over randomly shuffled filenames vs equal sampling from each class|bool|`False`|
|`eval_threshold`|Threshold above which to consider a prediction positive for evaluation|float|`0.5`|
|`epochs_lr_update`|Maximum number of epochs without improvement used to reset/decrease learning rate|int|`20`|
