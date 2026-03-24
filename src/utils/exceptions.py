class FraudDetectionException(Exception):
    """Base exception for Fraud Detection System"""
    pass

class DataGenerationError(FraudDetectionException):
    """Raised when there is an error generating simulated data"""
    pass

class PreprocessingError(FraudDetectionException):
    """Raised when there is an error preprocessing data"""
    pass

class ModelExecutionError(FraudDetectionException):
    """Raised when the model encounters an error during inference"""
    pass
