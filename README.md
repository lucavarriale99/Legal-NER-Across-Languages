# Towards Legal-NER Across Languages: Multilingual Framework and Evaluation

Legal Named Entity Recognition (L-NER) presents unique challenges due to the complex legal language and varied structures of legal documents. This study adapts NER models to the legal context, addressing the additional complexity in a multi-language domain. By employing dedicated models like various Legal BERTs and LUKEs, tailored for both monolingual and multilingual tasks, along with a novel multilingual dataset, we develop a comprehensive NER model for analyzing legal documents across languages. The research evaluates model performance on language-specific datasets, explores model embeddings using t-SNE visualizations, and assesses the effectiveness of multilingual adaptation. 
Results indicate promising performance from multilingual LUKE, particularly on a unified dataset, demonstrating its adaptability to diverse legal contexts. The t-SNE visualizations reveal distinct patterns in entity categorization among different models, providing insights into their interpretability and performance nuances. This research offers valuable insights for advancing NER models in the legal domain, emphasizing the importance of datasets and labels in multilingual adaptation.

For further information, please refer to the [paper](https://github.com/lucavarriale99/Legal-NER-Across-Languages/blob/main/paper.pdf)

Pablo Borrelli: <s303878@studenti.polito.it>

Daniele Mansillo: <s319297@studenti.polito.it>

Umberto Picone: <s296496@studenti.polito.it>

Luca Varriale: <s300795@studenti.polito.it>

## Repository structure
The repository is divided in the following folders:
- datasets: contains all the datasets used for our experiments<br>
The folder includes one sub-folder for each language (english, spanish, german and multilingual)
    - source: inside the folder source are located the scripts used to convert the various datasets to the same format as the english dataset
- model-exploration-extension: contains the code developed and used for generating embeddings and generate the t-SNE analysis
-multilingual-extension: contains the code developed and used for training models over the different datasets and for producing inferences exploiting the trained models

## How to run
The project can be reproduced by running the various jupyter notebooks found in the extensions' folders.
For each notebook we have provided two different versions, one meant to be run locally and one to be used by importing the single notebook on Google Colab (recommended approach).

### Local run
To run locally it is importanto to manually download from the links below the trained models and embeddings and copying them into the project folders before running the notebooks.

### Colab run (recommended)
To run on Colab it is possible to import the single notebooks both as a file or by cloning from the github repository.
Once imported in Colab, the notebooks will automatically loadd all the required dependencies and run the experiments (note: google drive has a treshold for consecutive downloads and automatic download may fail. In that case the data should be manually downloaded and loaded onto Colab)

### Scripts available
#### model-exploration-extension
- **colab_main.ipynb** : this notebook has the functionalities to train the models and generate embeddings, perform inference and finally perform the t-SNE analysis (version to be run on Colab)
- **local_main.ipynb** : this notebook has the functionalities to train the models and generate embeddings, perform inference and finally perform the t-SNE analysis (version to be run locally)

#### multilingual-extension
- **colab_training.ipynb** : this notebook trains language specific and multilingual models over the 4 datasets provided (version to be run on Colab)
- **local_training.ipynb** : this notebook trains language specific and multilingual models over the 4 datasets provided (version to be run locally)
- **colab_inference.ipynb** : this notebook, starting from the trained models performs inference for the spanish and englis languages version to be run on Colab)
- **local_inference.ipynb** : this notebook, starting from the trained models performs inference for the spanish and englis languages (version to be run locally)

## Source of trained models
All trained models are available for download through the following GDrive links
### Spanish dataset
PlanTL-GOB-ES/roberta-base-bne-capitel-ner: https://drive.google.com/drive/folders/1k0z09lszwJVw8isCC4IGqh2CXpkaMLcl?usp=sharing
studio-ousia/mluke-base: https://drive.google.com/drive/folders/1nPsA3b9UefwqP8VLQ0D-MC_872ETLbff?usp=sharing

### German dataset
elenanereiss/bert-german-ler: https://drive.google.com/drive/folders/1-OE8CrgLCJM3oJwBYIb_cMXYzdXNUhzI?usp=sharing
studio-ousia/mluke-base: https://drive.google.com/drive/folders/1JpXqeKrBk0a9aFGaMenlS4D9lncelMNP?usp=sharing

### Multilingual dataset
studio-ousia/mluke-base: https://drive.google.com/drive/folders/15CYs7VuaLQqgPxe1OXqyXYcPOVwBaWEe?usp=sharing

### English dataset
studio-ousia/mluke-base: https://drive.google.com/drive/folders/1w8anG8TREskpuNLiviYl0NZLNNxqlWv8?usp=sharing

### t-SNE embeddings
https://drive.google.com/drive/folders/1fcWeigGF6dLWdfyDsdQVBsQErpm-UeUp?usp=sharing

### English dataset trained models for t-SNE
https://drive.google.com/drive/folders/1-60uS-h64tC42ybk2zqac74iZFJt08so?usp=sharing
