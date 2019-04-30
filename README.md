## Assessing Incrementality in sequence-to-sequence models 

Repo containing the code for *Assessing Incrementality in sequence-to-sequence models* [TODO: Add link].

### Usage

To use the code in this repo, install all dependencies first:

    pip install -r requirements.txt
    
Models are then trained using the [machine](https://github.com/i-machine-think/machine) library (refer to the library's documentation for more information).

After finishing model training, you can use the following scripts inside the repo:

| Script        | Purpose        |
|:------------- |:-------------|
| ``evaluate.py`` | Evaluate a single model given some metrics |
| ``test_incrementality.py`` | Evaluate multiple models and compute metric scores per model type | 
| ``plot_correlation.py`` | Create correlation scatter plots and heat maps based on a list of models and metrics |
| ``qualitative_analysis.py`` | Perform a qualitative model analysis using the Integration Ratio |

The command line arguments of every individual script can be inspected using the `-h` or ``--help`` argument.


### Incremental Metrics

Metrics to measure the incremental processing capabilities of a model are defined in ``incremental_metrics.py`` and comprise
the following:

* **(Average) Integration Ratio**: Indicates whether the model prefers to integrate new information about the current input at every time step or is inclined to maintain a representation about the previous tokens.
* **Diagnostic Classifier Accuracy**: Quantifies to what extend information about previous tokens is contained within the current hidden representation.
* **Weighed Dianostic Classifier Accuracy**: The same as above, but models that are able to maintain information about inputs that 
occurred much earlier in the sequence are scored higher.
* **Representational Similarity**: Measure how much hidden representations after encoding the same subsequence of tokens 
resemble each other. Resemblance is quantified using a distance measure like cosine similarity or euclidean distance.
