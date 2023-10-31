# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py
from bigcode_eval.base import Task
from evaluate import load

class NewLibraries(Task):
    DATASET_PATH = "ml4se-group2/new-libraries-evaluation"
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            stop_words=["\n"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["prompt"]

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["reference"]
    
    def postprocess_generation(self, generation, idx):
        prompt = self.dataset["test"][idx]['prompt']
        gen = generation[len(prompt):]
        result = gen.split('\n', 1)[0]
        result.strip()
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        return result

    def process_results(self, generations, references):
        gens = [gen[0] for gen in generations]

        exact_match_metric = load("exact_match")
        exact_match = exact_match_metric.compute(predictions=gens, references=references)

        google_bleu_metrics = load("google_bleu")
        google_bleu = google_bleu_metrics.compute(predictions=gens, references=references)
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        return {"Exact Match" : exact_match, "Google BLEU" : google_bleu}
