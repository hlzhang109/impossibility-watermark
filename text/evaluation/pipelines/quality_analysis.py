# ===============================================
# quality_analysis.py
# Description: Pipeline for text quality analysis
# ===============================================

from tqdm import tqdm
from enum import Enum, auto
from watermark.base import BaseWatermark
from evaluation.dataset import BaseDataset
from evaluation.tools.text_editor import TextEditor
from exceptions.exceptions import InvalidTextSourceModeError, InvalidDirectAnalyzerTypeError, InvalidReferencedAnalyzerTypeError
from evaluation.tools.text_quality_analyzer import (TextQualityAnalyzer, DirectTextQualityAnalyzer, ReferencedTextQualityAnalyzer, 
                                                    ExternalDiscriminatorTextQualityAnalyzer)


class QualityPipelineReturnType(Enum):
    """Return type of the text quality analysis pipeline."""
    FULL = auto()
    SCORES = auto()
    MEAN_SCORES = auto()


class TextQualityComparisonResult:
    """Result of text quality comparison."""

    def __init__(self, watermarked_text: str, unwatermarked_text: str, 
                 watermarked_quality_score: float, unwatermarked_quality_score) -> None:
        """
            Initialize the text quality comparison result.

            Parameters:
                watermarked_text (str): The watermarked text.
                unwatermarked_text (str): The unwatermarked text.
                watermarked_quality_score (float): The quality score of the watermarked text.
                unwatermarked_quality_score (float): The quality score of the unwatermarked text.
        """
        self.watermarked_text = watermarked_text
        self.unwatermarked_text = unwatermarked_text
        self.watermarked_quality_score = watermarked_quality_score
        self.unwatermarked_quality_score = unwatermarked_quality_score
        pass


class TextQualityAnalysisPipeline:
    """Pipeline for text quality analysis."""

    def __init__(self, dataset: BaseDataset, 
                 watermarked_text_editor_list: list[TextEditor] = [],
                 unwatermarked_text_editor_list: list[TextEditor] = [], 
                 analyzer: TextQualityAnalyzer = None, unwatermarked_text_source='natural', 
                 show_progress: bool = True, return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        """
            Initialize the text quality analysis pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for evaluation.
                watermarked_text_editor_list (list[TextEditor]): The list of text editors for watermarked text.
                unwatermarked_text_editor_list (list[TextEditor]): The list of text editors for unwatermarked text.
                analyzer (TextQualityAnalyzer): The quality analyzer for text.
                unwatermarked_text_source (str): The source of unwatermarked text.
                show_progress (bool): Whether to show progress.
                return_type (QualityPipelineReturnType): The return type of the pipeline.
        """
        # Validate text_source_mode
        if unwatermarked_text_source not in ['natural', 'generated']:
            raise InvalidTextSourceModeError(unwatermarked_text_source)
        
        self.dataset = dataset
        self.watermarked_text_editor_list = watermarked_text_editor_list
        self.unwatermarked_text_editor_list = unwatermarked_text_editor_list
        self.analyzer = analyzer
        self.unwatermarked_text_source = unwatermarked_text_source
        self.show_progress = show_progress
        self.return_type = return_type
        pass

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        pass

    def _get_progress_bar(self, iterable):
        """Return an iterable possibly wrapped with a progress bar."""
        if self.show_progress:
            return tqdm(iterable, desc="Processing", leave=True)
        return iterable
    
    def _get_watermarked_text(self, watermark: BaseWatermark, index: int):
        """Generate watermarked text from dataset."""
        return watermark.generate_watermarked_text(self.dataset.get_prompt(index))
    
    def _get_unwatermarked_text(self, watermark: BaseWatermark, index: int):
        """Generate or retrieve unwatermarked text from dataset."""
        if self.unwatermarked_text_source == 'natural':
            return self.dataset.get_natural_text(index)
        elif self.unwatermarked_text_source == 'generated':
            return watermark.generate_unwatermarked_text(self.dataset.get_prompt(index))
    
    def _edit_watermarked_text(self, text: str, prompt: str = None):
        """Edit watermarked text using text editors."""
        for text_editor in self.watermarked_text_editor_list:
            text = text_editor.edit(text, prompt)
        return text
    
    def _edit_unwatermarked_text(self, text: str, prompt: str = None):
        """Edit unwatermarked text using text editors."""
        for text_editor in self.unwatermarked_text_editor_list:
            text = text_editor.edit(text, prompt)
        return text
    
    def _prepare_input_for_quality_analyzer(self, watermarked_text: str, unwatermarked_text: str, index: int):
        """Prepare input for quality analyzer."""
        pass

    def analyze_quality(self, prepared_data):
        """Analyze quality of watermarked and unwatermarked text."""
        pass

    def evaluate(self, watermark: BaseWatermark):
        """Conduct evaluation utilizing the pipeline."""
        evaluation_result = []
        bar = self._get_progress_bar(self._get_iterable())

        for index in bar:
            # Get watermarked and unwatermarked text
            watermarked_text = self._get_watermarked_text(watermark, index)
            unwatermarked_text = self._get_unwatermarked_text(watermark, index)

            # Edit watermarked and unwatermarked text
            edited_watermarked_text = self._edit_watermarked_text(watermarked_text, self.dataset.get_prompt(index))
            edited_unwatermarked_text = self._edit_unwatermarked_text(unwatermarked_text, self.dataset.get_prompt(index))

            # Prepare input for quality analyzer
            prepared_data = self._prepare_input_for_quality_analyzer(edited_watermarked_text, edited_unwatermarked_text, index)

            # Analyze quality of watermarked and unwatermarked text
            watermarked_quality_score, unwatermarked_quality_score = self.analyze_quality(prepared_data)

            # Append result
            evaluation_result.append(TextQualityComparisonResult(edited_watermarked_text, edited_unwatermarked_text, 
                                                                watermarked_quality_score, unwatermarked_quality_score))

        # Return result
        if self.return_type == QualityPipelineReturnType.FULL:
            return evaluation_result
        elif self.return_type == QualityPipelineReturnType.SCORES:
            return [{'watermarked': result.watermarked_quality_score, 
                     'unwatermarked': result.unwatermarked_quality_score} for result in evaluation_result]
        elif self.return_type == QualityPipelineReturnType.MEAN_SCORES:
            return {'watermarked': sum([result.watermarked_quality_score for result in evaluation_result]) / len(evaluation_result), 
                    'unwatermarked': sum([result.unwatermarked_quality_score for result in evaluation_result]) / len(evaluation_result)}


class DirectTextQualityAnalysisPipeline(TextQualityAnalysisPipeline):
    """
        Pipeline for direct text quality analysis.
    
        This class analyzes the quality of texts by directly comparing the characteristics of watermarked texts with unwatermarked texts. 
        It evaluates metrics such as perplexity (PPL) and log diversity without the need for any external reference text.
    
        Use this pipeline to assess the impact of watermarking on text quality directly.
    """

    def __init__(self, dataset: BaseDataset, 
                 watermarked_text_editor_list: list[TextEditor] = [], 
                 unwatermarked_text_editor_list: list[TextEditor] = [],
                 analyzer: TextQualityAnalyzer = None, unwatermarked_text_source='generated', 
                 show_progress: bool = True, return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        """
            Initialize the direct text quality analysis pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for evaluation.
                watermarked_text_editor_list (list[TextEditor]): The list of text editors for watermarked text.
                unwatermarked_text_editor_list (list[TextEditor]): The list of text editors for unwatermarked text.
                analyzer (TextQualityAnalyzer): The quality analyzer for text.
                unwatermarked_text_source (str): The source of unwatermarked text.
                show_progress (bool): Whether to show progress.
                return_type (QualityPipelineReturnType): The return type of the pipeline.
        """
        # Validate analyzer
        if not isinstance(analyzer, DirectTextQualityAnalyzer):
            raise InvalidDirectAnalyzerTypeError

        super().__init__(dataset, watermarked_text_editor_list, unwatermarked_text_editor_list, 
                         analyzer, unwatermarked_text_source, show_progress, return_type)
        pass

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.prompt_nums)
    
    def _prepare_input_for_quality_analyzer(self, watermarked_text: str, unwatermarked_text: str, index: int):
        """Prepare input for quality analyzer."""
        return watermarked_text, unwatermarked_text
    
    def analyze_quality(self, prepared_data):
        """Analyze quality of watermarked and unwatermarked text."""
        watermarked_text = prepared_data[0]
        unwatermarked_text = prepared_data[1]
        return self.analyzer.analyze(watermarked_text), self.analyzer.analyze(unwatermarked_text)


class ReferencedTextQualityAnalysisPipeline(TextQualityAnalysisPipeline):
    """
        Pipeline for referenced text quality analysis.

        This pipeline assesses text quality by comparing both watermarked and unwatermarked texts against a common reference text. 
        It measures the degree of similarity or deviation from the reference.
        
        Ideal for scenarios where the impact of watermarking on text quality needs to be assessed, particularly in relation to specific downstream tasks.
    """

    def __init__(self, dataset: BaseDataset, 
                 watermarked_text_editor_list: list[TextEditor] = [], 
                 unwatermarked_text_editor_list: list[TextEditor] = [],
                 analyzer: TextQualityAnalyzer = None, unwatermarked_text_source='generated', 
                 show_progress: bool = True, return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        """
            Initialize the referenced text quality analysis pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for evaluation.
                watermarked_text_editor_list (list[TextEditor]): The list of text editors for watermarked text.
                unwatermarked_text_editor_list (list[TextEditor]): The list of text editors for unwatermarked text.
                analyzer (TextQualityAnalyzer): The quality analyzer for text.
                unwatermarked_text_source (str): The source of unwatermarked text.
                show_progress (bool): Whether to show progress.
                return_type (QualityPipelineReturnType): The return type of the pipeline.
        """
        # Validate analyzer
        if not isinstance(analyzer, ReferencedTextQualityAnalyzer):
            raise InvalidReferencedAnalyzerTypeError
        super().__init__(dataset, watermarked_text_editor_list, unwatermarked_text_editor_list, 
                         analyzer, unwatermarked_text_source, show_progress, return_type)
        pass

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.prompt_nums)
    
    def _prepare_input_for_quality_analyzer(self, watermarked_text: str, unwatermarked_text: str, index: int):
        """Prepare input for quality analyzer."""
        return watermarked_text, unwatermarked_text, self.dataset.get_reference(index)
    
    def analyze_quality(self, prepared_data):
        """Analyze quality of watermarked and unwatermarked text."""
        watermarked_text = prepared_data[0]
        unwatermarked_text = prepared_data[1]
        reference = prepared_data[2]
        return self.analyzer.analyze(watermarked_text, reference), self.analyzer.analyze(unwatermarked_text, reference)


class ExternalDiscriminatorTextQualityAnalysisPipeline(TextQualityAnalysisPipeline):
    """
        Pipeline for external discriminator-based text quality analysis.
    
        This class utilizes an external discriminator, such as GPT-4, to compare the quality of watermarked and unwatermarked texts. 
        The discriminator evaluates the quality of the texts based on task descriptions provided by users, indicating potential degradation or preservation of quality due to watermarking.
    
        This analyzer is particularly useful when you need an advanced, AI-based opinion on the subtle impacts of watermarking.
    """

    def __init__(self, dataset: BaseDataset, 
                 watermarked_text_editor_list: list[TextEditor] = [], 
                 unwatermarked_text_editor_list: list[TextEditor] = [],
                 analyzer: TextQualityAnalyzer = None, unwatermarked_text_source='generated', 
                 show_progress: bool = True, return_type: QualityPipelineReturnType = QualityPipelineReturnType.MEAN_SCORES) -> None:
        """
            Initialize the external discriminator-based text quality analysis pipeline.

            Parameters:
                dataset (BaseDataset): The dataset for evaluation.
                watermarked_text_editor_list (list[TextEditor]): The list of text editors for watermarked text.
                unwatermarked_text_editor_list (list[TextEditor]): The list of text editors for unwatermarked text.
                analyzer (TextQualityAnalyzer): The quality analyzer for text.
                unwatermarked_text_source (str): The source of unwatermarked text.
                show_progress (bool): Whether to show progress.
                return_type (QualityPipelineReturnType): The return type of the pipeline.
        """
        # Validate analyzer
        if not isinstance(analyzer, ExternalDiscriminatorTextQualityAnalyzer):
            raise InvalidReferencedAnalyzerTypeError
        super().__init__(dataset, watermarked_text_editor_list, unwatermarked_text_editor_list, 
                         analyzer, unwatermarked_text_source, show_progress, return_type)
        pass

    def _get_iterable(self):
        """Return an iterable for the dataset."""
        return range(self.dataset.prompt_nums)
    
    def _prepare_input_for_quality_analyzer(self, watermarked_text: str, unwatermarked_text: str, index: int):
        """Prepare input for quality analyzer."""
        return watermarked_text, unwatermarked_text, self.dataset.get_prompt(index)
    
    def _score_for_judgement(self, judgement):
        """Return score based on judgement."""
        if judgement == 1:
            return 1, 0
        elif judgement == 2:
            return 0, 1
        return 0.5, 0.5

    def analyze_quality(self, prepared_data):
        """Analyze quality of watermarked and unwatermarked text."""
        watermarked_text = prepared_data[0]
        unwatermarked_text = prepared_data[1]
        prompt = prepared_data[2]
        judgement = self.analyzer.analyze(watermarked_text, unwatermarked_text, prompt)
        return self._score_for_judgement(judgement)
