# =============================================
# detection.py
# Description: Pipeline for watermark detection
# =============================================

from tqdm import tqdm
from enum import Enum, auto
from watermark.base import BaseWatermark
from evaluation.dataset import BaseDataset
from evaluation.tools.text_editor import TextEditor
from exceptions.exceptions import InvalidTextSourceModeError
import os, json

class DetectionPipelineReturnType(Enum):
    """Return type of the watermark detection pipeline."""
    FULL = auto()
    SCORES = auto()
    IS_WATERMARKED = auto()


class WatermarkDetectionResult:
    """Result of watermark detection."""

    def __init__(self, generated_or_retrieved_text, edited_text, detect_result) -> None:
        """
            Initialize the watermark detection result.

            Parameters:
                generated_or_retrieved_text: The generated or retrieved text.
                edited_text: The edited text.
                detect_result: The detection result.
        """
        self.generated_or_retrieved_text = generated_or_retrieved_text
        self.edited_text = edited_text
        self.detect_result = detect_result
        pass


class WatermarkDetectionPipeline:
    """Pipeline for watermark detection."""

    def __init__(self, dataset: BaseDataset, text_editor_list: list[TextEditor] = [], 
                 show_progress: bool = True, return_type: DetectionPipelineReturnType = DetectionPipelineReturnType.SCORES) -> None:
        """
            Initialize the watermark detection pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for the pipeline.
                text_editor_list (list[TextEditor]): The list of text editors.
                show_progress (bool): Whether to show progress bar.
                return_type (DetectionPipelineReturnType): The return type of the pipeline.
        """
        self.dataset = dataset
        self.text_editor_list = text_editor_list
        self.show_progress = show_progress
        self.return_type = return_type
       
    def _edit_text(self, text: str, prompt: str = None, watermark: BaseWatermark = None):
        """Edit text using text editors."""
        for text_editor in self.text_editor_list:
            text = text_editor.edit(text, prompt, watermark=watermark)
        return text
    
    def _generate_or_retrieve_text(self, dataset_index: int, watermark: BaseWatermark):
        """Generate or retrieve text from dataset."""
        pass

    def _detect_watermark(self, text: str, watermark: BaseWatermark):
        """Detect watermark in text."""
        detect_result = watermark.detect_watermark(text, return_dict=True)
        return detect_result

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        pass

    def _get_progress_bar(self, iterable):
        """Return an iterable possibly wrapped with a progress bar."""
        if self.show_progress:
            return tqdm(iterable, desc="Processing", leave=True)
        return iterable

    def evaluate(self, watermark: BaseWatermark, attack_name: str=""):
        """Conduct evaluation utilizing the pipeline."""
        evaluation_result = []
        bar = self._get_progress_bar(self._get_iterable())
        generated_or_retrieved_texts = []
        edited_texts = []

        for index in bar:
            generated_or_retrieved_text = self._generate_or_retrieve_text(index, watermark)
            edited_text = self._edit_text(generated_or_retrieved_text, self.dataset.get_prompt(index), watermark=watermark)
            detect_result = self._detect_watermark(edited_text, watermark)
            evaluation_result.append(WatermarkDetectionResult(generated_or_retrieved_text, edited_text, detect_result))
            generated_or_retrieved_texts.append(generated_or_retrieved_text)
            edited_texts.append(edited_text)
            print(f"Detected {index}-th examples. Score: {detect_result['score']}, \
                    Is watermarked: {detect_result['is_watermarked']}")
        
            if len(self.text_editor_list) != 0: 
                wtmk_detection_path = self.dataset.watermark_source.replace("watermarked.json", "watermakred_detection.json")
                with open(wtmk_detection_path, 'w') as f:
                    for i, text in enumerate(generated_or_retrieved_texts):
                        raw_text_detected_result = self._detect_watermark(text, watermark)
                        f.write(json.dumps({'id': index, 'prompt': self.dataset.get_prompt(i), 
                                            'watermarked_text': text, 'score': raw_text_detected_result['score'],
                                            'is_watermarked': raw_text_detected_result['is_watermarked']}) + '\n')
                print(f"Watermarked texts are saved to {wtmk_detection_path}")
            
            attack_path = self.dataset.watermark_source.replace("watermarked.json", f"{attack_name}.json")
            if len(self.text_editor_list) !=0 and attack_name != "": # not os.path.exists(attack_path) and 
                with open(attack_path, 'w') as f:
                    for i, text in enumerate(edited_texts):
                        f.write(json.dumps({'id': index, 'prompt': self.dataset.get_prompt(i), 
                                            'attacked_text': text, 'score': detect_result['score'],
                                            'is_watermarked': detect_result['is_watermarked'], 'watermarked_text': generated_or_retrieved_texts[i]
                                            }) + '\n')
                print(f"Attacked texts are saved to {attack_path}")

        if self.return_type == DetectionPipelineReturnType.FULL:
            return evaluation_result
        elif self.return_type == DetectionPipelineReturnType.SCORES:
            return [result.detect_result['score'] for result in evaluation_result]
        elif self.return_type == DetectionPipelineReturnType.IS_WATERMARKED:
            return [result.detect_result['is_watermarked'] for result in evaluation_result]


class WatermarkedTextDetectionPipeline(WatermarkDetectionPipeline):
    """Pipeline for detecting watermarked text."""

    def __init__(self, dataset, text_editor_list=[],
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES, *args, **kwargs) -> None:
        super().__init__(dataset, text_editor_list, show_progress, return_type)

    def _get_iterable(self):
        """Return an iterable for the prompts."""
        return range(self.dataset.prompt_nums)
    
    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Generate watermarked text from the dataset."""
        prompt = self.dataset.get_prompt(dataset_index)
        if len(self.dataset.watermarked_texts) > dataset_index:
            return self.dataset.watermarked_texts[dataset_index]
        else:
            return watermark.generate_watermarked_text(prompt)


class UnWatermarkedTextDetectionPipeline(WatermarkDetectionPipeline):
    """Pipeline for detecting unwatermarked text."""

    def __init__(self, dataset, text_editor_list=[], text_source_mode='natural',
                 show_progress=True, return_type=DetectionPipelineReturnType.SCORES, *args, **kwargs) -> None:
        # Validate text_source_mode
        if text_source_mode not in ['natural', 'generated']:
            raise InvalidTextSourceModeError(text_source_mode)
        
        super().__init__(dataset, text_editor_list, show_progress, return_type)
        self.text_source_mode = text_source_mode

    def _get_iterable(self):
        """Return an iterable for the natural texts or prompts."""
        if self.text_source_mode == 'natural':
            return range(self.dataset.natural_text_nums)
        else:
            return range(self.dataset.prompt_nums)
    
    def _generate_or_retrieve_text(self, dataset_index, watermark):
        """Retrieve unwatermarked text from the dataset."""
        if self.text_source_mode == 'natural':
            return self.dataset.get_natural_text(dataset_index)
        else:
            prompt = self.dataset.get_prompt(dataset_index)
            return watermark.generate_unwatermarked_text(prompt)
    